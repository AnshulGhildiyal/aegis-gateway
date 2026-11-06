"""Bidirectional NLI reranking backed by a quantized ONNX model."""

import asyncio
import json
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime  # type: ignore[import-untyped]
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from aegis.storage.cache import CacheEntry

_MODEL_REPOSITORY = "Xenova/nli-deberta-v3-small"
_MODEL_REVISION = "6bc2a55c7c0f7e2bc68de60bb248e523e2612abb"
_MAX_CANDIDATES = 2
_MODEL_MAX_LENGTH = 512


class _OnnxInput(Protocol):
    """Input metadata exposed by ONNX Runtime."""

    name: str


class _OnnxSession(Protocol):
    """Typed portion of the ONNX Runtime session interface used here."""

    def get_inputs(self) -> list[_OnnxInput]:
        """Return model input metadata."""

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        """Run model inference."""


class NliScorer(Protocol):
    """Score premise/hypothesis pairs with entailment probabilities."""

    async def entailment_probabilities(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        """Return the entailment probability for every pair."""

    async def close(self) -> None:
        """Release model execution resources."""


class OnnxNliScorer:
    """Run a pinned, quantized DeBERTa NLI model using ONNX Runtime CPU."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        max_length: int = _MODEL_MAX_LENGTH,
        max_workers: int = 1,
    ) -> None:
        if not 3 <= max_length <= _MODEL_MAX_LENGTH:
            raise ValueError(f"max_length must be between 3 and {_MODEL_MAX_LENGTH}")
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._cache_dir = cache_dir
        self._max_length = max_length
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="aegis-nli",
        )
        self._session: _OnnxSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._entailment_index: int | None = None
        self._inference_lock = asyncio.Lock()

    async def entailment_probabilities(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        """Score premise/hypothesis pairs off the event loop."""
        if not pairs:
            return []
        async with self._inference_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                self._score_sync,
                pairs,
            )

    async def close(self) -> None:
        """Shut down the inference executor."""
        await asyncio.to_thread(self._executor.shutdown, wait=True)

    def _score_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        self._load_model()
        if (
            self._session is None
            or self._tokenizer is None
            or self._entailment_index is None
        ):
            raise RuntimeError("NLI model failed to initialize")

        encodings = [
            self._tokenizer.encode(premise, hypothesis) for premise, hypothesis in pairs
        ]
        input_values: dict[str, np.ndarray] = {
            "input_ids": np.asarray(
                [encoding.ids for encoding in encodings],
                dtype=np.int64,
            ),
            "attention_mask": np.asarray(
                [encoding.attention_mask for encoding in encodings],
                dtype=np.int64,
            ),
            "token_type_ids": np.asarray(
                [encoding.type_ids for encoding in encodings],
                dtype=np.int64,
            ),
        }
        inputs = {
            item.name: input_values[item.name]
            for item in self._session.get_inputs()
            if item.name in input_values
        }
        missing_inputs = {
            item.name for item in self._session.get_inputs()
        } - inputs.keys()
        if missing_inputs:
            raise RuntimeError(
                f"ONNX model has unsupported inputs: {sorted(missing_inputs)}"
            )

        outputs = self._session.run(None, inputs)
        if not outputs:
            raise RuntimeError("ONNX model returned no outputs")
        logits = np.asarray(outputs[0], dtype=np.float64)
        if logits.ndim != 2 or logits.shape[0] != len(pairs):
            raise RuntimeError(
                "ONNX model output must have one row of logits per input pair"
            )
        if self._entailment_index >= logits.shape[1]:
            raise RuntimeError("Entailment label index is outside the model output")
        if not np.isfinite(logits).all():
            raise RuntimeError("ONNX model returned non-finite logits")

        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted_logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return [
            float(probabilities[row, self._entailment_index])
            for row in range(len(pairs))
        ]

    def _load_model(self) -> None:
        if self._session is not None:
            return
        cache_dir = str(self._cache_dir) if self._cache_dir is not None else None
        model_path = hf_hub_download(
            repo_id=_MODEL_REPOSITORY,
            filename="onnx/model_quantized.onnx",
            revision=_MODEL_REVISION,
            cache_dir=cache_dir,
        )
        tokenizer_path = hf_hub_download(
            repo_id=_MODEL_REPOSITORY,
            filename="tokenizer.json",
            revision=_MODEL_REVISION,
            cache_dir=cache_dir,
        )
        config_path = hf_hub_download(
            repo_id=_MODEL_REPOSITORY,
            filename="config.json",
            revision=_MODEL_REVISION,
            cache_dir=cache_dir,
        )
        with Path(config_path).open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        self._entailment_index = self._read_entailment_index(config)

        tokenizer = Tokenizer.from_file(tokenizer_path)
        pad_id = tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            raise RuntimeError("NLI tokenizer is missing its [PAD] token")
        tokenizer.enable_truncation(max_length=self._max_length)
        tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        self._tokenizer = tokenizer
        self._session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

    @staticmethod
    def _read_entailment_index(config: object) -> int:
        if not isinstance(config, dict):
            raise RuntimeError("NLI model config must be a JSON object")
        label2id = config.get("label2id")
        if isinstance(label2id, dict):
            for label, index in label2id.items():
                if (
                    isinstance(label, str)
                    and label.lower() == "entailment"
                    and isinstance(index, int)
                    and not isinstance(index, bool)
                    and index >= 0
                ):
                    return index
        id2label = config.get("id2label")
        if isinstance(id2label, dict):
            for index, label in id2label.items():
                if isinstance(label, str) and label.lower() == "entailment":
                    try:
                        parsed_index = int(index)
                    except ValueError:
                        continue
                    if parsed_index >= 0:
                        return parsed_index
        raise RuntimeError("NLI model config has no entailment label")


@dataclass(frozen=True, slots=True)
class NliDecision:
    """Bidirectional NLI scores and acceptance for one cache candidate."""

    candidate: CacheEntry
    candidate_to_query: float
    query_to_candidate: float
    accepted: bool


class NliReranker:
    """Apply a strict bidirectional entailment threshold to at most two candidates."""

    def __init__(
        self,
        scorer: NliScorer | None = None,
        *,
        entailment_threshold: float,
    ) -> None:
        if (
            not math.isfinite(entailment_threshold)
            or not 0 <= entailment_threshold <= 1
        ):
            raise ValueError("entailment_threshold must be between 0 and 1")
        self._scorer = scorer or OnnxNliScorer()
        self._threshold = entailment_threshold

    async def rerank(
        self,
        query: str,
        candidates: Sequence[CacheEntry],
    ) -> list[NliDecision]:
        """Score only the two highest-cosine candidates in both directions."""
        top_candidates = sorted(
            candidates,
            key=lambda candidate: candidate.similarity,
            reverse=True,
        )[:_MAX_CANDIDATES]
        if not top_candidates:
            return []

        pairs: list[tuple[str, str]] = []
        for candidate in top_candidates:
            pairs.extend(
                [
                    (candidate.user_content, query),
                    (query, candidate.user_content),
                ]
            )
        probabilities = await self._scorer.entailment_probabilities(pairs)
        if len(probabilities) != len(pairs):
            raise RuntimeError(
                f"NLI scorer returned {len(probabilities)} scores; "
                f"expected {len(pairs)}"
            )
        if any(
            not math.isfinite(probability) or not 0 <= probability <= 1
            for probability in probabilities
        ):
            raise RuntimeError("NLI scorer returned an invalid probability")

        decisions: list[NliDecision] = []
        for index, candidate in enumerate(top_candidates):
            candidate_to_query = probabilities[index * 2]
            query_to_candidate = probabilities[index * 2 + 1]
            decisions.append(
                NliDecision(
                    candidate=candidate,
                    candidate_to_query=candidate_to_query,
                    query_to_candidate=query_to_candidate,
                    accepted=(
                        candidate_to_query >= self._threshold
                        and query_to_candidate >= self._threshold
                    ),
                )
            )
        return decisions

    async def close(self) -> None:
        """Release scorer resources."""
        await self._scorer.close()
