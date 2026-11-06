# Aegis architecture

These diagrams describe the intended gateway flow. In Phase 0, only the HTTP API
exists; cache lookup, upstream forwarding, and circuit-breaker behavior are not
implemented yet.

## Request and response flow

```mermaid
sequenceDiagram
    participant Client
    participant Aegis
    participant Cache
    participant Upstream
    Client->>Aegis: POST /v1/chat/completions
    Aegis->>Cache: Check eligible request
    alt Cache hit
        Cache-->>Aegis: Cached completion
        Aegis-->>Client: Completion response
    else Cache miss or bypass
        Aegis->>Upstream: Forward request
        Upstream-->>Aegis: Completion response
        Aegis->>Cache: Store eligible response
        Aegis-->>Client: Completion response
    end
```

## Circuit breaker state machine

```mermaid
stateDiagram-v2
    [*] --> CLOSED
    CLOSED --> OPEN: Failure threshold reached
    CLOSED --> CLOSED: Successful request
    OPEN --> HALF_OPEN: Recovery timeout elapsed
    HALF_OPEN --> CLOSED: Probe succeeds
    HALF_OPEN --> OPEN: Probe fails
```

## Components

```mermaid
flowchart LR
    Client -->|OpenAI-compatible HTTP| API["FastAPI API"]
    API --> Engine["Request engine"]
    Engine --> Cache["Semantic cache"]
    Engine --> Breaker["Circuit breaker"]
    Engine --> Providers["Provider adapters"]
    Cache --> VectorStore["Vector store"]
    Cache --> Similarity["Similarity and NLI checks"]
    Providers --> LLM["LLM upstream"]
    Engine --> Storage["Storage adapters"]
```
