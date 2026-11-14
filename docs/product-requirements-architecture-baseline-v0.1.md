# Aegis Product Requirements and Architecture Baseline v0.1

**Status:** Approved product direction; detailed targets and open decisions are
identified below.  
**Purpose:** Establish the desired product and architecture baseline. This is
not a description of currently implemented or achieved behavior.

## 1. Product vision

Aegis is a self-hostable, fault-tolerant LLM gateway that helps organizations
serve successful inference tasks at lower total cost without compromising
answer quality, tenant isolation, or operational control. It provides an
OpenAI-compatible entry point and progressively adds exact reuse, quality-gated
semantic reuse, model routing, and output validation.

### Principles

- Correctness and privacy take priority over cache hit rate.
- A cache hit is acceptable only when request context is compatible and the
  response passes the configured quality policy.
- Make total cost per successful task—not request count or cache-hit rate—the
  primary optimization measure.
- Fail safely and make degraded behavior visible.
- Keep tenant and project boundaries explicit throughout execution and storage.
- Start with a simple self-hosted deployment while keeping service boundaries
  suitable for future managed-cloud operation.
- Measure outcomes; do not present targets, estimates, or historical
  evaluations as achieved product results.

### Goals

- Provide a dependable, authenticated gateway for the supported OpenAI API
  subset.
- Reduce inference spend while maintaining the strict quality policy.
- Let operators control provider credentials, cache eligibility, retention,
  and tenant/project access.
- Provide clear behavior during provider, cache, or storage failures.
- Build an architecture that can progress from persistent single-node storage
  to a managed, multi-tenant service.

### Non-goals for the initial release

- Full OpenAI API parity or support for every provider-specific parameter.
- Streaming responses; streaming is deferred.
- Multi-turn semantic caching; initial cache eligibility is limited to
  requests with a supported, single-turn context and explicitly established
  cache eligibility.
- Automatic model routing or escalation before its policy and quality gates
  are validated.
- A managed-cloud service in the first self-hosted release.
- Relaxing quality thresholds automatically to increase cache reuse.

## 2. Decision status and baseline

“Locked” means the product direction is approved. It does not mean implemented.
“Proposed target” means a planning objective that must be validated and may
require an explicit product decision before release. “Assumption” means a
planning premise that still needs confirmation.

| Area | Status | Baseline |
|---|---|---|
| Deployment | **Locked** | Self-hosted first, cloud-ready architecture, with future managed-cloud support. |
| API | **Locked** | Support an OpenAI-compatible subset, initially `POST /v1/chat/completions`. Streaming is deferred. |
| Identity | **Locked** | Instance → Tenant/Organization → Projects/Workspaces → API keys. Initially authenticate with API keys; resolve tenant identity from trusted server-side credentials; enforce strict isolation; provider credentials are administrator-managed. |
| Inference | **Locked** | Deterministic execution of gateway code/rules (not a guarantee of deterministic provider-generated output) → exact cache → quality-gated semantic cache → adaptive model routing/escalation → output validation. Optimize total cost per successful task. |
| Data | **Locked** | Metadata-only operational logging. Cache-eligible requests are cacheable by default, subject to tenant/project controls and a per-request opt-out evaluated before any cache interaction. Begin with persistent single-node storage. |
| Quality and reliability | **Locked** | Balanced product profile and strict quality policy. The 20% inference-cost reduction is an evaluation target; 99.9% availability is a design objective. Neither is an achieved result or guarantee. |
| Retention | **Proposed defaults** | Cache: 1 hour; metrics: 30 days; audit records: 90 days. Validate against deployment, compliance, and operator needs before release. |
| Initial runtime | **Assumption** | A single self-hosted instance can use persistent local/single-node storage; deployment sizing and supported operating environments remain to be specified. |

## 3. Functional requirements

### 3.1 API contract

- **API-01:** Expose `POST /v1/chat/completions` as the initial client-facing
  operation.
- **API-02:** Accept and return the supported OpenAI-compatible request and
  response subset. Publish the exact field and error contract before release;
  reject unsupported or malformed input explicitly rather than silently
  changing request meaning.
- **API-03:** Authenticate requests with an API key. Invalid, missing, revoked,
  or out-of-scope credentials must not reach a provider.
- **API-04:** Return provider and gateway failures in a documented,
  OpenAI-compatible error shape where possible, with an appropriate HTTP status.
- **API-05:** Streaming is outside the initial API contract. Do not imply
  streaming support until it has a separate implementation and acceptance
  criteria.

### 3.2 Identity and administration

- **ID-01:** Represent the hierarchy as instance, tenant/organization,
  project/workspace, and API key.
- **ID-02:** Derive tenant and project scope from trusted credential
  configuration. Do not treat caller-supplied tenant headers or body fields as
  authoritative identity.
- **ID-03:** Enforce tenant and project scope for cache reads, writes,
  administration, and usage/operational records. No credential may access
  another tenant’s data.
- **ID-04:** Allow authorized administrators to create, scope, rotate, and
  revoke API keys and configure provider credentials. Provider secrets must not
  be returned in API responses or written to operational logs.
- **ID-05:** Keep administrator responsibilities and any future role-based
  permission model explicit; the detailed roles and administrative API remain
  an open decision.

### 3.3 Inference and response quality

- **INF-01:** Execute supported requests deterministically where provider and
  request semantics permit, without treating low temperature as proof of
  determinism. Preserve all output-affecting parameters when forwarding,
  caching, or routing; only claim deterministic execution when the provider,
  model identity, request semantics, and generation settings support it.
- **INF-02:** Check exact reusable responses before semantic retrieval.
- **INF-03:** Apply semantic reuse only after strict context matching and all
  enabled quality gates pass.
- **INF-04:** Introduce adaptive model routing/escalation only when an explicit
  policy can choose a destination and evaluate the result against the strict
  quality policy.
- **INF-05:** Validate output against the request’s required format and other
  applicable constraints. Invalid output must not be reported as a successful
  validated result.
- **INF-06:** Evaluate optimization using total cost per successful task,
  including gateway/cache/verifier work and the cost of retries or escalation,
  rather than cache-hit rate alone.

## 4. Cache correctness, privacy, and deletion

### 4.1 Eligibility and opt-out

- **CACHE-01:** Determine cache eligibility from the supported request
  semantics, single-turn context, understood output-affecting generation
  parameters, absence of side effects, stable provider/model identity, and
  explicit cache policy. Temperature may be one eligibility constraint but
  does not establish determinism by itself. Multi-turn, unsupported, ambiguous,
  side-effecting, or otherwise ineligible requests bypass cache lookup and
  storage. Exclude dynamic and time-sensitive requests from both exact and
  semantic cache reuse unless an explicit freshness policy establishes that
  reuse is safe. Semantic similarity and a non-expired TTL do not independently
  establish freshness.
- **CACHE-02:** Treat cache eligibility as enabled by default for eligible
  requests, subject to tenant/project policy. A per-request opt-out must be
  honored before embedding, retrieval, cache access, single-flight locking, or
  cache writes. An opt-out request still follows the normal upstream path.
- **CACHE-03:** Streaming requests are excluded while streaming is deferred.
- **CACHE-04:** Do not log prompts or completions as operational metadata. Cache
  content is distinct from logs: storing a request and response is permitted
  only for eligible, opted-in requests under the configured retention and
  isolation policies.

### 4.2 Keying and semantic acceptance

- **CACHE-05:** Exact cache identity and semantic candidate filters must include
  the tenant and resolved project scope, system-prompt hash, embedding
  model/version, target LLM, temperature, `max_tokens`, tools, and
  `response_format`. Preserve every other output-affecting request parameter in
  the strict key or reject it from caching until its semantics are defined.
- **CACHE-06:** Generate semantic embeddings from user content only. Apply
  exact metadata filters before considering a candidate; never allow a
  cross-tenant, cross-project, or incompatible-context candidate to proceed to
  quality checks.
- **CACHE-07:** Semantic acceptance has three gates: (1) cosine vector retrieval
  on user content with exact payload filters; (2) a Stage-0 regex guard that
  compares numbers and negations; and (3) bidirectional NLI, checking both
  entailment directions, on only the top one or two candidates. A candidate
  that fails any required gate is a miss.
- **CACHE-08:** The same strict quality policy governs cache hits in every
  operating mode, including degraded mode. No mode or configuration may
  silently lower the acceptance standard. If no result satisfies the policy,
  return the documented failure for that condition rather than an insufficiently
  validated answer.
- **CACHE-09:** Coalesce identical concurrent requests only by exact strict
  cache key plus identical payload, using per-worker single-flight behavior.
  Never coalesce by semantic similarity. Streaming is excluded.

### 4.3 Retention and deletion

- **DATA-01:** Enforce configurable retention for cache entries, metrics, and
  audit records. The proposed initial defaults are 1 hour, 30 days, and 90
  days, respectively; these are not finalized compliance commitments.
- **DATA-02:** Support deletion or invalidation scoped at least to the relevant
  tenant and, where applicable, project. Deletion must cover exact and
  semantic indexes, payloads, and associated cache metadata, not just hide
  entries from ordinary reads.
- **DATA-03:** Define and document deletion completion semantics, including
  backups and replicas (when introduced). Deletion must also prevent
  pre-existing in-flight writes from recreating deleted entries; define and
  test the completion boundary before offering a managed-cloud service.
- **DATA-04:** Keep operational logs metadata-only. Limit metadata to what is
  necessary for diagnosis, usage accounting, and audit; exclude credentials,
  prompts, completions, and provider secrets. Document any unavoidable
  sensitive metadata and its retention.
- **CACHE-10:** Invalidate or make unreachable cache entries when an output-
  affecting provider/model identity, embedding or verifier version, cache
  policy, or other relevant compatibility version changes. Define which
  changes require invalidation and verify that stale entries cannot pass the
  current compatibility filters.
- **CACHE-11:** Coordinate deletion with in-flight lookups and writes so an
  operation started before deletion cannot make deleted content available
  afterward. Specify the deletion barrier/ordering semantics in the storage
  design and test races.
- **CACHE-12:** Serve a stale response only during an eligible degraded-mode
  request and only when its age, measured from the original generation time,
  does not exceed an explicitly configured finite maximum stale age. Do not
  serve stale data when that limit is absent or exceeded. Exclude dynamic and
  time-sensitive requests from stale serving. Require exact tenant and resolved
  project scope, current request/model/policy compatibility, and the same
  strict quality validation required for every cache hit. Semantic similarity
  or bidirectional NLI entailment does not establish freshness. Fresh-cache
  TTL and maximum stale age are separate limits; retain expired entries only
  as long as needed for bounded stale eligibility, then delete them. Include
  response metadata identifying degraded service and the response's age/source
  without exposing prompts, completions, or secrets; define the exact
  age/source representation in the API contract. The exact API response
  schema/header representation is an unresolved API-contract decision. The
  numeric maximum stale age and any default remain unresolved product
  decisions.

## 5. Non-functional requirements and objectives

### Performance and cost

- **NFR-PERF-01:** Measure end-to-end latency by outcome (exact hit, semantic
  hit, upstream miss, retry, and degraded response) and report distributions
  under a documented workload and deployment configuration.
- **NFR-PERF-02:** Establish latency and capacity service objectives from
  repeatable baselines before production claims. No latency target is currently
  approved. For benchmark planning only, 50 ms cache-hit p95 and 100 ms
  cache-miss gateway-overhead p95 may be evaluated as provisional engineering
  targets; they are not locked commitments or achieved results. Benchmark
  conditions, including workload, deployment, warmup, and the boundary of
  gateway overhead, must be specified before comparing results.
- **NFR-COST-01:** Evaluate the balanced profile against a measured baseline
  with **20% total inference cost reduction as an evaluation target**, while
  preserving strict quality acceptance. Report methodology and quality/cost
  trade-offs; do not claim the target is met without measured evidence.

### Quality

- **NFR-QUALITY-01:** Maintain a strict semantic cache quality policy. Measure
  false cache hits as a first-class risk, along with precision, recall, and
  bypass behavior on representative and adversarial evaluations.
- **NFR-QUALITY-02:** Do not equate embedding similarity, offline benchmark
  performance, or a cache hit with safe answer reuse.
- **NFR-QUALITY-03:** Preserve the original upstream result when validation
  fails only where the API contract permits; otherwise return a clear failure
  rather than silently substituting an unvalidated answer.

### Security and privacy

- **NFR-SEC-01:** Enforce authentication and authorization on every protected
  operation, and isolate cache and administrative access at tenant and project
  boundaries.
- **NFR-SEC-02:** Protect API keys and provider credentials in storage and
  transit; support credential rotation/revocation and prevent secret exposure
  in responses, logs, metrics, and errors.
- **NFR-SEC-03:** Use an explicit secret-management approach for API keys and
  provider credentials, covering provisioning, access control, rotation,
  revocation, and removal. Do not expose provider credentials to untrusted
  request-controlled input or tenant scope.
- **NFR-SEC-04:** Apply least privilege to service components and storage
  access. Define secure defaults and operator guidance for network exposure,
  TLS, persistent data, and backups.
- **NFR-SEC-05:** Restrict outbound provider calls to administrator-configured
  provider destinations; use authenticated, TLS-protected connections with
  certificate validation. Do not allow caller-controlled destinations or
  redirects to bypass outbound security policy.
- **NFR-SEC-06:** Provide a way to disable caching at request, project, and
  tenant scope. Verify that per-request opt-out occurs before any cache
  interaction.
- **NFR-SEC-07:** Enforce configurable limits for request size, generation
  budgets, concurrent work, and queued work. Reject or shed work explicitly
  when limits are reached; exact limits and per-tenant quotas remain deployment
  and capacity-design decisions.

### Reliability and operations

- **NFR-REL-01:** Design for a **99.9% availability objective**. This is a
  design objective, not a current SLO, measured availability, or guarantee.
  Define the measurement window, eligible requests, and exclusions before
  claiming compliance.
- **NFR-REL-02:** Bound retries and request time spent waiting on providers.
  Detect provider availability failures, including timeouts and 5xx responses,
  and account for one exhausted retry chain as one provider failure. Treat
  rate limiting (429) separately: it is not an availability failure or
  circuit-breaker failure, and retry behavior must respect provider guidance
  such as `Retry-After` within configured bounds.
- **NFR-REL-03:** Provide circuit-breaker behavior with CLOSED, OPEN, and
  HALF-OPEN states. Provider availability failures drive the breaker toward
  OPEN; OPEN suppresses routine provider attempts and uses eligible safe
  fallback behavior; HALF-OPEN permits bounded recovery probes; a successful
  probe returns to CLOSED and a failed probe returns to OPEN. Exact thresholds,
  transition implementation, retry timing/backoff algorithm, and clock/test
  mechanism belong in a future reliability design specification.
- **NFR-REL-04:** In OPEN/degraded mode, stale cache responses are eligible only
  under the explicit freshness and isolation requirements in CACHE-12 and only
  when they pass the same strict quality policy as every other cache hit. Mark
  degraded responses with `X-Aegis-Degraded: true`. If no acceptable result is
  available, return a documented failure (503 when the provider is unavailable)
  with degraded-response metadata; never return an insufficiently validated
  answer.
- **NFR-REL-05:** Failure of optional cache infrastructure must safely bypass
  caching and allow core inference to proceed where possible. If coordination
  or single-flight infrastructure fails, use a documented safe local strategy
  where possible; otherwise proceed without coalescing rather than making core
  inference depend on optional optimization infrastructure.
- **NFR-REL-06:** Support request deadlines and cancellation. Propagate
  cancellation to provider work where supported, stop work at the applicable
  deadline, release coordination resources, and do not cache partial or
  cancelled results.
- **NFR-REL-07:** Expose health and operational signals sufficient to
  distinguish provider availability failures, rate limits, cache/coordination
  failures, degraded responses, and normal cache misses without logging prompts
  or completions.
- **NFR-REL-08:** Bound telemetry cardinality and avoid sensitive or
  request-specific values in metric labels. Do not use prompts, completions,
  credentials, raw API keys, or unbounded tenant/project/request identifiers
  as metric dimensions; document any controlled dimensions and their privacy
  implications.
- **NFR-REL-09:** For future streaming support, pass through upstream events
  while assembling chunks; never cache a partial response after client
  disconnect; replay cache hits as SSE. Streaming requires a separate release
  acceptance gate and is not part of the initial release.

## 6. Logical architecture and responsibilities

This is the desired logical architecture, not a claim that each component
exists today. Initial deployment may package several responsibilities in one
process; boundaries should remain testable and replaceable.

| Responsibility | Owns |
|---|---|
| API edge | OpenAI-compatible request/response subset, validation, status/error mapping, request opt-out parsing, and authentication handoff. |
| Identity and policy | Trusted API-key resolution; instance/tenant/project scope; provider credential access; cache, retention, and routing policy. |
| Request orchestrator | Eligibility decisions and the ordered inference path; coordinates cache, provider, reliability, and output validation without embedding storage/provider-specific logic. |
| Exact cache | Strict-key lookup/write, TTL, scoped invalidation, and exact-hit metadata. |
| Semantic cache engine | User-content embeddings, filtered vector retrieval, Stage-0 numeric/negation guard, bidirectional NLI verification, and accepted-candidate selection. |
| Provider adapters | Provider-specific request/response translation, timeouts, retry handling, and normalized errors. Credentials are supplied through administrator-managed configuration. |
| Routing and escalation | Policy-based model selection and escalation after the earlier stages are validated; records cost and outcome metadata needed for evaluation. |
| Output validation | Format/constraint validation and clear invalid-output behavior. |
| Reliability controls | Provider failure classification, bounded retry and deadline policy, circuit-breaker behavior, degraded-mode decision, and safe fallback behavior. |
| Storage adapters | Persistent single-node storage initially; abstractions for cache/vector storage and later distributed services. |
| Telemetry and audit | Metadata-only operational signals and scoped audit records with configured retention. |
| Administration | API-key lifecycle, tenant/project policy, provider credential management, retention, and deletion controls. |

### Request path

1. Authenticate the API key and resolve trusted tenant/project scope.
2. Validate the supported request subset and resolve opt-out and eligibility
   before any cache-related work.
3. Apply deterministic gateway rules and attempt strict exact lookup; this
   does not guarantee deterministic provider-generated output.
4. On an eligible exact miss, run filtered semantic retrieval and the required
   quality gates.
5. On a miss or bypass, call the configured provider through reliability
   controls; adaptive routing/escalation applies only when enabled by an
   explicit policy.
6. Validate the output; store only eligible, opted-in, successful results under
   the strict key and retention policy.
7. Emit metadata-only operational signals. On failure, apply breaker/degraded
   behavior without returning an unverified stale answer.

## 7. Phased roadmap and release acceptance

The roadmap describes desired sequencing, not authorization to begin a later
implementation phase. Each phase requires its own scope approval.

### Phase 0 — Baseline and API contract

- Freeze the supported API subset, identity resolution rules, configuration
  model, error contract, and deployment assumptions.
- **Accept when:** contract tests cover valid/invalid authentication,
  malformed/unsupported input, tenant resolution, error responses, and
  per-request opt-out semantics; documentation identifies unimplemented
  behavior.

### Phase 1 — Deterministic forwarding and exact cache

- Implement the authenticated single-turn request path, provider adapter,
  understood request semantics and generation settings, strict exact cache,
  persistence, and scoped deletion.
- **Accept when:** exact reuse requires a complete compatible key; incompatible
  tenants/projects and output-affecting parameters never share entries;
  opt-out and ineligible requests perform no cache interaction; deletion and
  retention tests cover exact data; provider failures are explicit.

### Phase 2 — Quality-gated semantic cache and resilience

- Add user-content-only retrieval, metadata filters, Stage-0, bidirectional
  NLI, exact-key single-flight, circuit breaker, and safe degraded mode.
- **Accept when:** tests demonstrate each gate rejects unsafe candidates,
  semantic retrieval cannot cross scope or key filters, single-flight requires
  identical payloads, retry chains count once, 429 does not count as a
  provider availability failure, circuit-breaker states and bounded recovery
  are validated, and stale responses meet freshness, isolation, strict-quality,
  and degraded-metadata requirements.
- **Accept when:** documented, repeatable quality evaluation reports false
  cache hits and trade-offs; no benchmark threshold or claimed result is
  substituted for this evidence.

### Phase 3 — Adaptive routing and output validation

- Add explicit model selection/escalation and response validation, then
  evaluate cost per successful task and strict-quality outcomes.
- **Accept when:** routing and escalation are deterministic under configured
  policy, output validation has positive and negative contract tests, all
  cost-accounted stages are included in evaluation, and measured results
  support or reject the 20% target without weakening quality policy.

### Phase 4 — Production operations and managed-cloud readiness

- Harden deployment, observability, backup/recovery, multi-instance storage,
  and tenant administration; pursue managed-cloud support only after
  self-hosted requirements are met.
- **Accept when:** the availability objective has a defined measurement
  contract and supporting operational evidence; recovery, deletion, isolation,
  credential lifecycle, and retention are tested; security and deployment
  reviews are complete. Passing this gate is not implied by design intent.

### Deferred streaming release

Streaming is separate from the initial API release. **Accept when:** SSE
pass-through and cache-hit replay conform to the published contract; client
disconnect tests prove partial responses are not cached; cancellation,
provider failure, and resource cleanup are covered; and streaming requests
remain excluded from semantic cache/single-flight unless a later decision
changes that policy.

## 8. Historical repository context

Existing repository documentation is evidence of prior work and evaluation,
not the specification above. [architecture.md](./architecture.md) describes a
gateway flow and circuit-breaker diagram but also says that only the HTTP API
exists in “Phase 0.” [phase1_summary.md](./phase1_summary.md) records work on
embeddings, retrieval, Stage-0, NLI, and evaluation, while [results.md](./results.md)
contains dataset-specific experimental reports and their limitations. These
documents have different scopes and phase labels; neither should be read as a
complete current-state audit or as proof that this baseline's production
requirements are implemented.

Historical benchmark results must remain tied to their dataset, split,
configuration, and run methodology. They do not establish production quality,
cost reduction, latency, or availability. Update implementation status and
measured evidence separately from this requirements baseline.

## 9. Risks, assumptions, and unresolved decisions

### Risks

- A false semantic hit can be more costly than an upstream miss; benchmark
  similarity may not represent the cost or impact of wrong answer reuse.
- A strict quality policy may reduce cache reuse enough to offset inference
  savings. The quality/cost trade-off must be measured, not assumed.
- Provider behavior, throttling, and retry semantics vary and can affect both
  availability and cost.
- Persistent single-node storage creates local durability, backup, capacity,
  and availability responsibilities for self-hosted operators.
- Metadata can still be sensitive; “metadata-only” requires an explicit
  allowlist and retention policy.
- Multi-tenant operation increases the impact of key/filter/deletion defects.
- A 99.9% objective and future managed-cloud support require operational
  capabilities not established by the current design alone.

### Assumptions

- The initial customer deployment is operated by the customer in a
  self-hosted environment.
- API keys can be securely provisioned and mapped to trusted tenant/project
  scope outside request-controlled identity fields.
- Initial workloads can be served without streaming and without semantic
  reuse of multi-turn conversations.
- Administrators can manage provider credentials and configure policies
  outside the inference request path.

### Unresolved decisions

- Exact supported request fields, provider list, model identifiers, and
  compatibility/versioning policy.
- API-key creation, storage, rotation, revocation, project assignment, and
  administrator authorization details.
- Semantic thresholds, verifier model/version, evaluation datasets, release
  false-hit tolerance, and how thresholds are selected for the strict policy.
- Exact cache-eligibility policy for stochastic provider settings, side-effecting
  requests, unrecognized output-affecting fields, and evidence needed to
  establish stable provider/model identity.
- Maximum allowed stale age, default freshness policy, and the request classes
  that must be treated as dynamic or time-sensitive.
- Routing/escalation policy, supported validation constraints, and behavior
  when the selected model or validator is unavailable.
- Retention configurability, deletion guarantees for backups, and applicable
  privacy/compliance obligations.
- Performance and capacity targets; availability measurement window and
  exclusions; deployment support matrix and backup/recovery objectives.
- Reliability design details including retry budgets/timing, circuit-breaker
  transition policy, request deadlines, and coordination fallback behavior.
- Resource limits and quotas; outbound network policy; secret-management
  implementation; telemetry dimensions and cardinality controls.
- Cache invalidation rules for provider/model, embedding/verifier, and policy
  changes; deletion ordering and completion semantics for in-flight writes.
- Whether a distributed cache or coordination service is needed in each phase
  and the migration criteria from persistent single-node storage.
- Streaming release scope and acceptance thresholds.
