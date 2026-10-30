# SM120 NVFP4 Decode bucketing contribution plan

Status: FlashInfer module validation; vLLM runtime integration is pending and no
upstream pull request is open. This document is a working design on the
contribution branch, not a claim that the backend is implemented in this checkout.

## Preparation checkpoint (2026-10-30)

The source-level FlashInfer module is on the companion contribution branch:
[implementation commit](https://github.com/Nicko-F/flashinfer/commit/9c62c3bfb401bcb87cd265ed7fecfd4650557d41).
It includes optional native task-context hooks, a separate planner/workspace API,
package JIT/FFI bindings, changed-input Graph tests, and a matched native/bucket
module example. No runtime source rewriting or private path/hash gates remain in
this new implementation.

Source delivery has recovered. Source-package import, CUDA compilation and all
six focused FlashInfer tests passed on SM120, including changed-input Graph
replay. Seven matched two-layer timing windows show gains for skewed lengths,
but regressions for short and uniform inputs; see the companion module's
[validation table](https://github.com/Nicko-F/flashinfer/blob/feat/sm120-nvfp4-decode-bucketing/flashinfer/experimental/xqa_decode_buckets/README.md).
The example's explicit-device/seed fix is commit `71cb089` in that fork.

The vLLM runtime integration has not started. The initial
B=1/L=512 measurement was 34.986 us versus native 23.525 us for two
layers; aligned takes 25.973 us versus 23.517 us. The persistent planner assigns
both tiles to one active CTA, while native uses two. These are measured complete
windows; the follow-up component measurements are below.
This port retains native QK/PV and does not include the separate
Q6-to-N8 transposed Compute proposal. Do not attribute that proposal's potential
gain to the current implementation or claim universal improvement.

Follow-up diagnosis measured persistent compute alone at 32.454 us. Allowing
one-tile buckets restores two active owners and reduces compute to 23.722 us,
matching the same-run native 23.718 us. All six focused tests pass after the
change. The full plan-plus-compute window is still 26.589 us (+12.1%); this
remaining integration cost is unresolved, so no small-load speedup is claimed.
The reviewed correction is [FlashInfer commit d632c9c](https://github.com/Nicko-F/flashinfer/commit/d632c9c4541956fccf68773b4db187b8b5668a79).

The following checkpoint, [FlashInfer ab14b59](https://github.com/Nicko-F/flashinfer/commit/ab14b59656feb7ebcba606021fbd0571a2f192e7),
caps the fixed grid by static page-table capacity and derives B=1 task geometry
from live device length without assignment. Eight focused test groups passed,
including changing lengths, zero/restore and page updates under Graph replay.
It measured about 3.01x for live length512 with table capacity128K; tight B1
still regressed 2.7-3.1%, and four short requests still regressed 17.8%.

The direct path now supports the structural condition
`B == 1 or B * ceil(table_width * 32 / 256) <= SMs`. For the second case, every
potential tile already fits the available CTA budget: direct `(tile, request)`
grid coordinates remove assignment and task-list traversal. Longer multi-request
work retains the shared interval plan. Eligibility depends only on static shape
and device properties, so it cannot change silently during Graph replay; no
device-to-host length read or tuned load threshold is introduced. Both paths
retain native XQA arithmetic and merge.

The extended module passed all eight focused test groups. Matched two-layer
complete Graph windows are 21.469 us for B1/256, 23.822 us for B1/512 and 23.874 us
for four 512-token requests, versus paired native means of 21.261/23.629/23.650 us.
Residual regressions are 0.8-1.0%, not a universal non-regression pass. B1 with a
512-token live prefix and 128K table capacity measures 23.310 us versus 70.822 us
native (about 3.04x). Both sides retain the same table capacity. These are module
results, not vLLM serving results; actual framework table width matters. Wide
multi-request tables and prepared uniform workloads still need integration
coverage. The current direct kernel uses 186 registers and has zero compiler
spill loads/stores. See the companion module report for full paired results.
The tested extension is [FlashInfer commit 2d81984](https://github.com/Nicko-F/flashinfer/commit/2d819842ee8915b7bab623bcf77a1a575aacf132).

## Repository and branch policy

- vLLM integration: `Nicko-F/vllm:feat/sm120-nvfp4-decode-bucketing`
  targets `vllm-project/vllm:main`.
- Kernel dependency: `Nicko-F/flashinfer:feat/sm120-nvfp4-decode-bucketing`
  targets `flashinfer-ai/flashinfer:main`.
- Fork main branches track upstream; keep contribution commits on feature
  branches. Do not merge the research repository's history into either fork.
- Keep Prefill, pipeline parallelism, P/D transport and unrelated model kernels
  outside this contribution. Do not publish experimental raw directories here.

## Contribution boundary

The existing research implementation adapts native XQA through generated source
substitutions and a private ctypes/JIT wrapper. That is not the intended upstream
implementation. Replace the substitutions with explicit source-level task/range
interfaces, preserving the original native entry and native QK/PV mathematics.

FlashInfer owns device task assignment, persistent compute, partial storage and
merge, native bindings, compilation, capability checks and workspace API. Use its
existing JIT/FFI infrastructure. Its experimental-backend policy is a possible
initial landing path, subject to maintainer review; no acceptance is assumed.

vLLM owns opt-in configuration, capability-based dispatch and the metadata/workspace
lifecycle. Extend its existing FlashInfer backend rather than injecting project
worker monkeypatches or introducing another scheduler/framework.

## Dataflow and reuse

1. Build one interval task plan per compatible Decode metadata group per forward.
2. Reuse that plan across attention layers sharing the same request order and live
   KV lengths; do not assume a fixed number of layers.
3. Keep layer Q, KV and output pointers independent. Sequential layers may reuse
   scratch only after the previous compute/merge completes.
4. On Graph replay, stable buffer addresses hold refreshed live metadata. Planning
   must execute at the correct per-forward boundary, not once at capture time or
   independently for every layer.
5. Preserve native normalized BF16 partial output plus FP32 max/sum merge semantics,
   independent effective BMM1/BMM2 scales and proper PDL acquire/completion ordering.

An implementation with per-layer planning cannot inherit measurements from the
shared-plan research implementation. Count assignment, compute, merge and dispatch
costs in comparisons.

### Native integration boundary under review

The current vLLM `FlashInferMetadataBuilder.build()` creates Decode metadata once
per attention group. The worker shares that metadata across the group's layer
names. This is the intended plan-preparation boundary; the attention forward path
should only consume the prepared plan and its workspace.

The first integration should refresh stable device length buffers and prepare the
plan before replaying the model Graph. Do not retain a stale tensor view from the
first captured batch. Count the preparation launch and any metadata copy in model
timing: this boundary differs from the historical captured planning prelude.

Capture compatibility must cover changed request order, lengths, page IDs and
padding. Selection between persistent, aligned and native kernels must also be
compatible with the captured graph; changing a Python branch outside capture
does not change the kernels already recorded inside it. Decide that selection
after the source-level module measurements, rather than copying an unverified
research dispatch threshold.

This is an integration design, not an implemented or measured vLLM path yet.

### First integration decisions

Source review confirms that `GPUModelRunner._build_attention_metadata()` assigns
one builder result to every layer in its attention group. The initial integration
therefore stays inside the existing FlashInfer builder/implementation; it does not
add a worker scheduler or patch the model forward.

- Use an explicit attention configuration with native as the default and separate
  persistent/aligned experimental choices. Keep the selected compute path stable
  for each captured shape. Do not infer a dynamic routing policy from one timing
  point or change a captured kernel by changing a Python branch after capture.
- A builder owns cached plans and workspaces keyed by the compatible padded
  request count and page-table width. Each entry owns stable device lengths;
  refresh those from the current metadata, then prepare once before the model
  consumes them. Different builders/microbatches must not overwrite each other's
  pending plans. Sequential compatible layers may share scratch.
- Preserve the existing q_len=1 XQA padding convention: query/output views cover
  `num_decodes`, which may exceed `num_decode_tokens`. Zero-length padded rows
  remain zero-length. Test refresh after request reordering, length growth and
  removal, including returning to a previously captured batch size.
- Reuse `_nvfp4_kv_views()` and the existing effective BMM1/BMM2 scales. Its views
  split each page's contiguous data and scale regions; do not repack the cache or
  add an intermediate dequantization allocation. Unsupported geometry/features
  use the original native route. Missing experimental dependency support must
  be reported clearly when the option is requested.
- Prepare/JIT outside graph capture; attention forward only consumes the plan.
  Measure external length refresh and planning as part of the integration cost.
  Module graphs that capture planning internally are a different timing window.
- Extend the nearby FlashInfer NVFP4 attention tests using real builders and
  implementations. First validate two compatible layers and changing metadata in
  eager/Graph modes, then one real-model entry-point anchor. Full service curves
  come after this module integration is complete.

These are implementation decisions, not completed validation. The first module
measurements will determine whether either experimental choice is worth promoting.

## Initial supported scope

The first specialization may remain deliberately narrow: tested SM120 geometry,
BF16 Q/output, one query token per request, head dimension 256, Q/KV head ratio
6 with one local KV head, page size 32, packed NVFP4 KV and its block scales.
Sliding windows, sinks, soft caps, speculative decode and other unsupported layouts
retain the native path. These are tensor/kernel constraints, not a Qwen or TP4
identity check. Batch capacity and supported specialization must be explicit.

Retain the audited persistent kernel's actual thread/barrier and shared-memory
constraints. Do not widen its eligibility by merely deleting assertions. Derive
SM count and launch/workspace parameters from hardware and shapes; do not fix a
physical GPU ordinal or assume 170 SMs.

## Portability requirements

- No personal paths, environment names, run directories or mandatory receipts in
  library code. Discover package assets/cache through standard upstream mechanisms.
- No exact source-hash or Git-revision runtime gate. Check necessary API and layout
  contracts; record revisions only as provenance.
- Normal content-addressed JIT caching and ordinary dependency lockfiles remain
  valid mechanisms, not runtime access restrictions.
- Remove incidental 128K model limits and exactly-16-layer assumptions. Keep actual
  integer, allocation and supported-kernel bounds documented.
- Keep Graph capture sizes separate from kernel eligibility.
- Caller-owned workspaces must document concurrency and stream ordering.
- Keep local experiment admission and memory safeguards outside reusable kernels.

## Bounded migration sequence

1. Source-level FlashInfer implementation and public package binding. Preserve
   native default, add explicit opt-in, port nearby tests instead of creating a
   second test or execution framework.
2. Package build, numerical tolerance checks and changed-input Graph replay.
   Inspect resources; measure matched module latency including plan frequency.
3. Native vLLM metadata/dispatch integration, then one targeted real-model lifecycle
   and Graph anchor with native fallback. Do not run full service curves per patch.
4. At integration completion, run representative small, uniform and skewed Decode
   workloads through existing benchmarks and one matched service comparison.
   Report variation and tails; no universal speedup or bitwise-equivalence claim.

## Existing evidence versus work still required

Historical kernel, Graph and service tests motivate this contribution. They do not
validate the new upstream package build or new metadata lifecycle.

Reuse existing behavioral coverage for unique tile ownership, zero rows, page/tile
tails, nontrivial quantization scales, padding, scratch capacity, changed lengths,
Graph replay and fallback. Replace tests of source-string substitution with tests
of actual package behavior. Add binding mutation/fake/meta/opcheck coverage as
required by the selected upstream interface.

Before submission: check open issues/PRs for overlapping work; use the repository
PR template; run relevant lint and tests; document visible configuration and
supported hardware; retain original licensing/attribution; include AI assistance
and the human contributor's DCO signoff. The human submitter must review and
understand the patch. Do not present pending tests as passed.

## References

- [vLLM contribution policy](https://docs.vllm.ai/en/latest/contributing/)
- [vLLM agent contribution requirements](https://github.com/vllm-project/vllm/blob/main/AGENTS.md)
- [vLLM FlashInfer backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py)
- [FlashInfer contribution policy](https://github.com/flashinfer-ai/flashinfer/blob/main/CONTRIBUTING.md)
- [FlashInfer XQA JIT construction](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/jit/xqa.py)

Source revisions are recorded by Git history, not enforced by application code.
