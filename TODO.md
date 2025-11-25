# SafetyKnob — Intent, Review, and Action Plan

## Project Intent Summary

- Research motivation
  - Proactively prevent industrial accidents by detecting potential hazards in images.
  - Leverage pre-trained vision backbones (CLIP, SigLIP, DINOv2, EVA-CLIP) for broad generalization and data efficiency.
  - Provide interpretable, multi-dimensional safety assessment (fall, collision, equipment, environmental, protective gear) rather than a single binary label.

- Practical engineering goal
  - Build a configurable system that:
    - Extracts embeddings using one or more backbones and optionally caches them.
    - Trains a lightweight classifier to predict per-dimension scores and overall safety.
    - Combines models via ensembling (Weighted Vote, Stacking) to improve accuracy and robustness.
    - Exposes a CLI (assess/train/evaluate/compare/experiment) and a FastAPI server for real-time and batch inference.
    - Produces metrics and exports results (JSON/CSV) with optional reports/visualizations.

- Design principles
  - Configuration-first (config.json): models, checkpoints, dimensions, thresholds, training params.
  - Extensible: add models/dimensions easily; swap checkpoints without code changes.
  - Performant: GPU acceleration, caching, batch evaluation.
  - Observable: health/info endpoints, metrics, and reproducible benchmarks.


## Code vs. Intent Review

- Strong alignment
  - Orchestration: `SafetyAssessmentSystem` integrates embedders → per-model NN classifiers → ensembling (`weighted_vote`, `stacking`).
  - Config-driven: `SystemConfig.from_dict` maps `config.json` (models/dimensions/strategies/training) to runtime.
  - CLI/API: `main.py` provides the end-to-end workflow; `src/api/server.py` includes `health`, `info`, `assess`, and a working batch `assess/batch`.
  - Caching hooks: embedders support file-path keyed embedding caches.

- Gaps and inconsistencies
  - API docs vs implementation
    - Docs mention `/api/v1/models`, but server exposes `/api/v1/info` for model listing. Docs mark batch as “planned” while `/api/v1/assess/batch` is implemented.
  - Threshold usage
    - `SafetyAssessmentSystem.assess_image` uses a fixed `> 0.5` decision; it ignores `config.safety.safety_threshold` and `config.safety.confidence_threshold`.
  - Checkpoint handling
    - Embedders hardcode default checkpoints and ignore `checkpoint` in `config.json`; this breaks configurability.
    - CLIP id mismatch: code uses `openai/clip-vit-large-patch14-336` while `config.json` lists `openai/clip-vit-large-patch14`.
  - Class name collision
    - Two `SafetyClassifier` classes exist (NN in `src/core/neural_classifier.py`, legacy in `src/core/classifier.py`). `src/core/__init__.py` exports both → confusion risk.
  - Dimension scoring integration
    - `DimensionAnalyzer` (similarity-based) is instantiated but unused in inference; NN heads produce dimension outputs. Methodologies are not unified.
  - Evaluation inefficiency
    - `evaluate_dataset` recomputes embeddings multiple times (via `assess_image` then per-model loop). `ImageDataset` returns tensors unused by embedding code paths.
  - Ensemble naming
    - Inconsistent names: `weighted_vote` vs `weighted_voting`; `EnsembleConfig.method` references `weighted_voting` while other parts use `weighted_vote`.
  - API file handling/security
    - Temporary files saved as `/tmp/{filename}` can collide under concurrency; content-type/size checks are minimal; CORS is `*`; no auth/rate limiting.
  - Device configuration
    - Per-model `device` in `config.models` is ignored; a single device is applied to all embedders.
  - Error fallback and dims
    - `extract_single_embedding` returns a fixed 768-dim zero vector on failure, regardless of the model’s embedding dimension.


## Feedback (Research + Engineering Aligned)

- AI Research perspective
  - Clarify score semantics and thresholds
    - Define “safety score” vs “risk score” (training often uses `1 - risk`). Document and align code/metrics/visuals.
    - Select thresholds empirically (ROC/PR analysis); provide scripts that recommend global and per-dimension thresholds.
  - Probability calibration and uncertainty
    - Calibrate outputs (temperature scaling/Platt); report ECE/Brier. Tie “confidence” to calibrated probabilities.
    - Report ensemble agreement/variance/entropy and define human-review policies for low-consensus predictions.
  - Dimension scoring methodology
    - Support `dimension_scoring: nn | similarity | hybrid`. Compare modes on labeled subsets; use hybrid when labels are sparse/noisy.
  - Data and labels strategy
    - Provide a labeling guide with edge cases for the 5 dimensions; optionally bootstrap with CLIP text prompts (pseudo-labels).
  - Explainability
    - Add saliency (e.g., Grad-CAM) and per-dimension explanations in reports to aid prevention and debugging.

- Software Engineering perspective
  - Configuration fidelity
    - Honor `checkpoint` from `config.json`; verify `embedding_dim` at runtime (warn/error on mismatch).
    - Apply `safety_threshold`/`confidence_threshold` consistently; expose via CLI/API.
    - Decide on global vs per-model device; implement consistently and document.
  - Resolve legacy ambiguity
    - Rename legacy classifier (e.g., `SafetyClassifierLegacy`) and move to `src/legacy/` or clearly deprecate. Export NN classifier by default.
  - API robustness and security
    - Use `NamedTemporaryFile` or in-memory processing; unique names; validate size/type; standardize JSON errors. Optional API keys and rate limiting; tighten CORS in prod.
  - Evaluation efficiency
    - Cache embeddings per image per model within a run; avoid duplicate extraction. Align `ImageDataset` with embedding paths (prefer file paths when embeddings are needed).
  - Testing/CI and benchmarking
    - Unit tests (config mapping, thresholds, ensemble logic, API validation, MockEmbedder), small fixture dataset, reproducibility (seeds). Benchmark script with env metadata.


## Actionable TODOs

### High priority

- [ ] Apply thresholds from config
  - [ ] Use `config.safety.safety_threshold` (not fixed `0.5`) for `is_safe` in `src/core/safety_assessment_system.py:SafetyAssessmentSystem.assess_image`.
  - [ ] Define/apply `config.safety.confidence_threshold` semantics (e.g., annotate low-confidence or gate decisions) in outputs and evaluation.
  - [ ] Ensure CLI/API and `evaluate_dataset` reflect configured thresholds.

- [ ] Honor model checkpoints and embedding dims
  - [ ] Update `src/core/embedders.py` to accept and use per-model `checkpoint` from config (SigLIP/CLIP/DINOv2/EVA-CLIP).
  - [ ] Validate actual embedding dimension vs `config.models[i].embedding_dim` and warn/error on mismatch.
  - [ ] Align CLIP model id with config or derive `embedding_dim` from the loaded model.
  - [ ] Fix `extract_single_embedding` fallback: on failure, propagate error or return a zero vector sized to the model’s embedding dimension; log clearly.

- [ ] Resolve classifier naming and exports
  - [ ] Rename `src/core/classifier.py:SafetyClassifier` → `SafetyClassifierLegacy` (or move to `src/legacy/`).
  - [ ] Update `src/core/__init__.py` to export only the NN `SafetyClassifier` by default; export the legacy one under the new name if needed.

- [ ] API and docs consistency
  - [ ] Choose canonical model info endpoint: implement `/api/v1/models` or standardize on `/api/v1/info`; update `docs/API_REFERENCE.md` accordingly.
  - [ ] Update `docs/API_REFERENCE.md` to show `/api/v1/assess/batch` as supported with accurate request/response and error examples.
  - [ ] Unify ensemble naming to `weighted_vote` across code and docs; normalize `EnsembleConfig.method`.

- [ ] API robustness/security
  - [ ] Replace `/tmp/{filename}` with UUID `NamedTemporaryFile` or in-memory processing to avoid collisions; sanitize filenames.
  - [ ] Enforce file size/type limits; validate actual image content; return standardized JSON errors.
  - [ ] Add optional API key auth and simple rate limiting; document HTTPS and production CORS.

- [ ] Evaluation efficiency
  - [ ] Add an in-run embedding cache keyed by `(model_name, image_path)` to avoid re-extraction in `evaluate_dataset` and per-model loops.
  - [ ] Align `ImageDataset` with embedding paths (prefer file paths when embeddings are needed) to avoid double loading/transform.

### High priority (Current Progress - Model Training)

- [x] **SigLIP model training completed**
  - Training time: ~5 hours (18,249 seconds)
  - Test performance: Accuracy: 95.3%, F1: 95.5%, AUC: 99.1%
  - Dimension scores: protective_gear (84.8%), equipment_hazard (53.8%), collision_risk (44.5%), fall_hazard (41.0%), environmental_risk (35.5%)
  - Results saved to: `results/single_models/siglip/`

**Upcoming Automated Tasks** (to be executed sequentially after SigLIP training):
- [ ] CLIP model training
- [ ] DINOv2 model training
- [ ] Ensemble experiments (combining trained models)
- [ ] Research analysis report generation

### High priority (v2 track now in repo)

- [x] Add parallel v2 pipeline for safe benchmarking
  - [x] `src_v2/core/embedders_v2.py`: honors `checkpoint`, validates dims, safe fallback
  - [x] `src_v2/core/safety_assessment_system_v2.py`: uses config thresholds, in-run embedding cache, per-model device support
  - [x] `src_v2/core/__init__.py`, `src_v2/__init__.py`: expose v2 symbols
  - [x] `main_v2.py`: CLI wrapper to run v2 without touching v1
- [ ] Benchmark v1 vs v2 on representative data
  - [ ] Latency/throughput per model and ensemble
  - [ ] Accuracy/F1 on labeled subset; confirm threshold behavior
  - [ ] Memory footprint and model load time
- [x] Add benchmark script (does not modify v1)
  - [x] `scripts/benchmark_v2.py` with optional `--compare-v1`
  - [x] Outputs JSON summary to `results/`
- [ ] If benchmarks meet targets, proceed with Option B (archive + integrate)
  - [ ] Create `archive/YYYYMMDD/` snapshot of touched v1 files
  - [ ] Port v2 improvements into v1 files with minimal diffs
  - [ ] Update docs to reflect unified behavior
  - Note: Per latest decision, v1 is archival-only; continue evolving v2 exclusively until explicit approval to integrate.


### Medium priority

- [ ] Dimension scoring unification
  - [ ] Add `dimension_scoring` mode (`nn|similarity|hybrid`) to config and wire `SafetyAssessmentSystem` to use `DimensionAnalyzer`/NN heads accordingly.
  - [ ] Provide an experiment script to compare modes on labeled subsets; report per-dimension AUC/AP with plots.

- [ ] Probability calibration
  - [ ] Implement temperature scaling/Platt scaling after training; persist calibration params with checkpoints.
  - [ ] Report ECE/Brier; tie `confidence` to calibrated probabilities when enabled.

- [ ] Typed configs and validation
  - [ ] Introduce a typed `ModelConfig` for `config.models` (name, model_type, checkpoint, embedding_dim, device, cache_dir) with validation.
  - [ ] Normalize model_type values (e.g., `eva_clip` vs `evaclip`, `dinov2` vs `dino`) via a canonical map and validate inputs.
  - [ ] Decide/document per-model device vs global device; enforce consistently.

- [ ] Benchmarking and logging
  - [ ] Add `scripts/benchmark.py` to measure latency/throughput/memory per model and ensemble; persist results with GPU/software versions and config.
  - [ ] Add structured logging for latency/memory; make log level configurable.

- [ ] Testing and reproducibility
  - [ ] Unit tests: config parsing, threshold logic, ensemble weighting update, API validation (happy/error cases), MockEmbedder determinism.
  - [ ] Regression tests: small labeled fixture for end-to-end assess/evaluate; verify CSV/JSON outputs.
  - [ ] Set seeds for torch/numpy/python; document deterministic flags where feasible.

### Medium priority (repository hygiene and cleanup)

- [ ] Remove tracked build/cache/log/result artifacts from git and rely on existing .gitignore
  - [ ] Purge committed results files: `*_results_*.json`, `*_results_*.csv`, `*_summary_*.json`
  - [ ] Remove committed `logs/` contents
  - [ ] Remove committed `__pycache__/` directories and `*.pyc`
- [ ] Consider moving large dataset content out of repo (keep only minimal samples under `data/` or externalize)
- [ ] Remove `.claude/` (non-project artifact)
- [ ] Move legacy/unused modules instead of deleting
  - [ ] `src/core/classifier.py` → `src/legacy/core/classifier_legacy.py` (or similar)
  - [ ] `src/utils/analysis_utils.py` (if only used by legacy) → `src/legacy/utils/`
  - [ ] `src/api/inference.py` (legacy fallback) → `src/legacy/api/` (server-based API is primary)
- [ ] Align examples with current API
  - [ ] `examples/api_client.py` uses `/api/v1/models` — update to `/api/v1/info` or add `/api/v1/models` endpoint
- [ ] Relocate demo scripts for clarity
  - [ ] Move `demo.py`, `prepare_test_all.py`, `test_all_data.py`, `test_all_folder.py` under `examples/` or `scripts/` (they are not core libs)


### Low priority

- [ ] Cleanup and organization
  - [ ] Move legacy code (e.g., `src/core/classifier.py`, unused parts of `src/api/inference.py`) under `src/legacy/` or clearly deprecate.
  - [ ] Remove dead/duplicated ensemble code paths; keep a single canonical implementation.

- [ ] Embedding extraction quality and resilience
  - [ ] Handle EXIF orientation and corrupted images; improve exception messages in embedders.
  - [ ] Consider defaulting DINOv2 to a smaller backbone for practicality, with “giant” opt-in via config.
  - [ ] Provide offline mode guidance (local checkpoints, HF cache paths).

- [ ] Documentation
  - [ ] Provide a realistic `labels.json` example for the 5 dimensions (with edge cases and guidance).
  - [ ] Add “How thresholds are chosen” (ROC/PR + calibration plots) and a recommendation workflow.
  - [ ] Optional Dockerfile and env-var configuration for deployment; production checklist (HTTPS, API keys, rate limiting, CORS).
  - [x] Add `_ko` locale mirrors for Markdown docs (README_ko.md, docs/*_ko.md, TODO_ko.md)


## Implementation Notes and File Pointers

- Thresholds
  - `src/core/safety_assessment_system.py`: use `config.safety.safety_threshold` and `config.safety.confidence_threshold`; reflect in metrics/outputs.

- Embedders
  - `src/core/embedders.py`: accept/use `checkpoint`; verify `embedding_dim`; align CLIP id; handle EXIF/errors; fix zero-vector fallback sizing.
  - `create_embedder`/`get_embedder`: pass `checkpoint` and `cache_dir`; enforce canonical `model_type` mapping.

- Classifier naming/export
  - `src/core/classifier.py` (rename to legacy), `src/core/__init__.py` (export policy).

- Ensemble naming/logic
  - `src/core/ensemble.py`: unify naming to `weighted_vote`; ensure `update_weights` and stacking paths are consistent.

- API and docs
  - `src/api/server.py`: unique temp files or in-memory; optional `/api/v1/models`; stricter validation; standardized errors.
  - `docs/API_REFERENCE.md`: update endpoints, batch support, error schema; `README`/`DEVELOPMENT`: refresh performance numbers with benchmark results.
  - `main_v2.py`: run v2 pipeline side-by-side for benchmarking (`python main_v2.py …`).
  - `scripts/benchmark_v2.py`: run benchmarks without touching v1 (`--compare-v1` optional).
  - `src_v2/api/server_v2.py`: v2 FastAPI server with `/api/v1/models`, safe temp handling, and strict validation.

- Evaluation efficiency
  - `src/core/safety_assessment_system.py:evaluate_dataset`: in-run embedding cache; prevent duplicate extraction.
  - `src/utils/data_utils.py:ImageDataset`: align with embedding workflows (favor file paths when needed).


## Success Criteria

- Functional
  - Configured thresholds applied end-to-end; embedders load configured checkpoints; batch/single inference robust to concurrency; API/docs aligned.

- Research
  - Calibrated probabilities with uncertainty metrics; dimension scoring modes compared with plots; threshold selection script provides recommended values.

- Engineering
  - No naming collisions; deterministic tests pass; benchmark script reproduces README metrics; structured logs provide operational visibility.
