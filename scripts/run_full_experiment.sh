#!/usr/bin/env bash
# ==============================================================================
# SafetyKnob Full Experiment Reproduction Pipeline
# ==============================================================================
#
# This script reproduces ALL experiments for the SafetyKnob paper.
# Run from the project root directory: bash scripts/run_full_experiment.sh
#
# Prerequisites:
#   - Raw data in data/raw/danger_al/{safe,danger,caution}/
#   - Python 3.8+ with: torch, transformers, scikit-learn, matplotlib, seaborn, tqdm
#   - GPU recommended for Steps 2, 5, 6, 8 (embedding extraction + CNN baseline + DANN + multitask)
#   - Steps 3, 4, 7, 9, 10 run on CPU (probe training from cached embeddings)
#
# Estimated time:
#   - Steps 1 (data prep):         ~5 min  (CPU)
#   - Step 2 (embeddings):         ~2-4 hr (GPU, 3 models x 2 splits)
#   - Steps 3-4 (probes):          ~10 min (CPU)
#   - Step 5 (CNN baselines):      ~1-2 hr (GPU)
#   - Step 6 (DANN):               ~30 min (CPU from embeddings)
#   - Step 7 (category probes):    ~5 min  (CPU)
#   - Step 8 (multitask):          ~2-3 hr (GPU, on-the-fly extraction)
#   - Step 9 (ensemble ablation):  ~5 min  (CPU)
#   - Step 10 (scaling curve):     ~5 min  (CPU)
#   - Steps 11-12 (analysis):      ~2 min  (CPU)
# ==============================================================================

set -euo pipefail

# Colors for progress messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Timestamps
start_total=$(date +%s)

log_step() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] STEP $1: $2${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

log_substep() {
    echo -e "${YELLOW}  >> $1${NC}"
}

log_done() {
    local end=$(date +%s)
    local elapsed=$((end - $1))
    echo -e "${GREEN}  Done in ${elapsed}s ($((elapsed/60))m $((elapsed%60))s)${NC}"
}

SEEDS="42,123,456,789,2024"

# ==============================================================================
# STEP 1: Prepare Data Splits [CPU]
# ==============================================================================
log_step 1 "PREPARE DATA SPLITS [CPU]"
step_start=$(date +%s)

# 1a. Create scenario split (safe+danger+caution, 70/15/15 random stratified)
log_substep "1a. Creating scenario split (safe+danger+caution) -> data_scenario/"
python3 scripts/create_scenario_split.py

# 1b. Generate 5D labels from accident codes
log_substep "1b. Generating 5D safety labels -> data_scenario/labels_5d.json"
python3 scripts/generate_5d_labels.py

# 1c. Create temporal split (train: Jun-Sep 2022, test: Oct-Nov 2022)
log_substep "1c. Creating temporal split -> data_temporal/"
python3 scripts/create_temporal_split.py

# 1d. Create caution-excluded split (safe+danger only, 70/15/15)
log_substep "1d. Creating caution-excluded split -> data_caution_excluded/"
python3 scripts/create_split_exclude_caution.py

log_done "$step_start"

# ==============================================================================
# STEP 2: Extract Embeddings for All Models [GPU REQUIRED]
# ==============================================================================
log_step 2 "EXTRACT EMBEDDINGS [GPU REQUIRED]"
step_start=$(date +%s)

MODELS="siglip clip dinov2"
SPLITS="scenario temporal"

for split in $SPLITS; do
    for model in $MODELS; do
        log_substep "Extracting ${model} embeddings for ${split} split"
        python3 scripts/extract_embeddings.py \
            --model "$model" \
            --data-dir "data_${split}" \
            --output "embeddings/${split}/${model}"
    done
done

# Also extract embeddings for caution_excluded split
for model in $MODELS; do
    log_substep "Extracting ${model} embeddings for caution_excluded split"
    python3 scripts/extract_embeddings.py \
        --model "$model" \
        --data-dir "data_caution_excluded" \
        --output "embeddings/caution_excluded/${model}"
done

log_done "$step_start"

# ==============================================================================
# STEP 3: Run All Probe Experiments - 3 models x 3 probes x 5 seeds [CPU]
# ==============================================================================
log_step 3 "PROBE EXPERIMENTS (Scenario Split, RQ1) [CPU]"
step_start=$(date +%s)

for model in $MODELS; do
    for depth in linear 1layer 2layer; do
        for seed in 42 123 456 789 2024; do
            log_substep "${model}/${depth} seed=${seed}"
            python3 experiments/train_from_embeddings.py \
                --model "$model" \
                --embeddings-dir "embeddings/scenario/${model}" \
                --probe-depth "$depth" \
                --output "results/scenario/${model}_${depth}_seed${seed}" \
                --seed "$seed" \
                --epochs 50 \
                --save-predictions
        done
    done
done

log_done "$step_start"

# ==============================================================================
# STEP 4: Run Temporal Experiments (RQ2) [CPU]
# ==============================================================================
log_step 4 "TEMPORAL EXPERIMENTS (RQ2) [CPU]"
step_start=$(date +%s)

for model in $MODELS; do
    for seed in 42 123 456 789 2024; do
        log_substep "${model}/2layer (temporal) seed=${seed}"
        python3 experiments/train_from_embeddings.py \
            --model "$model" \
            --embeddings-dir "embeddings/temporal/${model}" \
            --probe-depth 2layer \
            --output "results/temporal/${model}_2layer_seed${seed}" \
            --seed "$seed" \
            --epochs 50 \
            --save-predictions
    done
done

log_done "$step_start"

# ==============================================================================
# STEP 5: Run Baseline CNN Experiments [GPU REQUIRED]
# ==============================================================================
log_step 5 "BASELINE CNN EXPERIMENTS [GPU REQUIRED]"
step_start=$(date +%s)

for cnn_model in resnet50 efficientnet_b0; do
    for seed in 42 123 456 789 2024; do
        # Finetuned (full backbone training)
        log_substep "${cnn_model} finetuned seed=${seed}"
        python3 experiments/train_baseline.py \
            --model "$cnn_model" \
            --data-dir data_scenario \
            --output "results/baseline/${cnn_model}_finetuned_seed${seed}" \
            --epochs 30 \
            --seed "$seed"

        # Frozen backbone (linear probe on CNN features)
        log_substep "${cnn_model} frozen seed=${seed}"
        python3 experiments/train_baseline.py \
            --model "$cnn_model" \
            --data-dir data_scenario \
            --output "results/baseline/${cnn_model}_frozen_seed${seed}" \
            --epochs 30 \
            --frozen \
            --seed "$seed"
    done
done

log_done "$step_start"

# ==============================================================================
# STEP 6: Run DANN Domain Adaptation [CPU from embeddings]
# ==============================================================================
log_step 6 "DANN DOMAIN ADAPTATION [CPU]"
step_start=$(date +%s)

for model in $MODELS; do
    log_substep "DANN ${model} (5 seeds)"
    python3 experiments/train_dann.py \
        --model "$model" \
        --source-embeddings "embeddings/scenario/${model}" \
        --target-embeddings "embeddings/temporal/${model}" \
        --output "results/dann/${model}" \
        --seeds "$SEEDS" \
        --epochs 100
done

log_done "$step_start"

# ==============================================================================
# STEP 7: Category-Specific Experiments (RQ3) [CPU]
# ==============================================================================
log_step 7 "CATEGORY-SPECIFIC EXPERIMENTS (RQ3) [CPU]"
step_start=$(date +%s)

for cat in A B C D E; do
    for seed in 42 123 456 789 2024; do
        log_substep "Category ${cat} (SigLIP/2layer) seed=${seed}"
        python3 experiments/train_from_embeddings.py \
            --model siglip \
            --embeddings-dir "embeddings/scenario/siglip" \
            --probe-depth 2layer \
            --category "$cat" \
            --output "results/category/siglip_2layer_cat${cat}_seed${seed}" \
            --seed "$seed" \
            --epochs 50 \
            --save-predictions
    done
done

log_done "$step_start"

# ==============================================================================
# STEP 8: Multi-Task 5D Experiment [GPU REQUIRED]
# ==============================================================================
log_step 8 "MULTI-TASK 5D EXPERIMENT [GPU REQUIRED]"
step_start=$(date +%s)

for model in $MODELS; do
    log_substep "Multi-task ${model}"
    python3 experiments/train_multitask.py \
        --model "$model" \
        --data-dir data_scenario \
        --labels-file data_scenario/labels_5d.json \
        --output "results/multitask/${model}" \
        --epochs 20 \
        --batch-size 32 \
        --seed 42
done

log_done "$step_start"

# ==============================================================================
# STEP 9: Ensemble Ablation [CPU]
# ==============================================================================
log_step 9 "ENSEMBLE ABLATION [CPU]"
step_start=$(date +%s)

log_substep "Running ensemble ablation on scenario split"
python3 scripts/run_ensemble_ablation.py \
    --embeddings-base embeddings \
    --split scenario \
    --output results/ensemble_ablation \
    --seeds "$SEEDS"

log_done "$step_start"

# ==============================================================================
# STEP 10: Data Scaling Curve [CPU]
# ==============================================================================
log_step 10 "DATA SCALING CURVE [CPU]"
step_start=$(date +%s)

log_substep "Running scaling curve (SigLIP 2-layer, 10-100% data)"
python3 scripts/run_scaling_curve.py \
    --embeddings-dir "embeddings/scenario/siglip" \
    --output results/scaling_curve \
    --seeds "$SEEDS"

log_done "$step_start"

# ==============================================================================
# STEP 11: Run Multi-Seed Summary & Statistical Tests [CPU]
# ==============================================================================
log_step 11 "MULTI-SEED SUMMARY & STATISTICAL TESTS [CPU]"
step_start=$(date +%s)

log_substep "Generating multi-seed summary (all RQs)"
python3 scripts/run_multiseed.py \
    --output results/multiseed \
    --embeddings-base embeddings \
    --experiments all \
    --seeds "$SEEDS"

log_done "$step_start"

# ==============================================================================
# STEP 12: Generate Paper Figures [CPU]
# ==============================================================================
log_step 12 "GENERATE PAPER FIGURES [CPU]"
step_start=$(date +%s)

# Comparison visualizations (scenario vs caution_excluded)
log_substep "Generating comparison visualizations"
python3 scripts/compare_experiments.py || echo "WARNING: compare_experiments.py requires results in results/scenario and results/caution_excluded"

# t-SNE visualization (requires raw images + GPU for embedding extraction)
log_substep "Generating t-SNE visualizations"
python3 scripts/visualize_tsne.py \
    --model siglip \
    --data-dir data_scenario \
    --output results/figures/tsne_siglip.png || echo "WARNING: t-SNE visualization failed (may need GPU)"

# Scaling curve plot is auto-generated in Step 10

log_done "$step_start"

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
end_total=$(date +%s)
elapsed_total=$((end_total - start_total))

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN} ALL EXPERIMENTS COMPLETE${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "Total time: ${elapsed_total}s ($((elapsed_total/3600))h $((elapsed_total%3600/60))m $((elapsed_total%60))s)"
echo ""
echo "Results structure:"
echo "  results/"
echo "  ├── scenario/          # RQ1: 3 models x 3 probes x 5 seeds"
echo "  ├── temporal/          # RQ2: 3 models x 2-layer x 5 seeds"
echo "  ├── category/          # RQ3: 5 categories x 5 seeds (SigLIP)"
echo "  ├── baseline/          # CNN baselines (ResNet-50, EfficientNet-B0)"
echo "  ├── dann/              # DANN domain adaptation (3 models x 5 seeds)"
echo "  ├── multitask/         # Multi-task 5D (3 models)"
echo "  ├── ensemble_ablation/ # Ensemble subset ablation"
echo "  ├── scaling_curve/     # Data scaling curve"
echo "  ├── multiseed/         # Combined multi-seed summary"
echo "  ├── comparison/        # Scenario vs Caution-excluded comparison"
echo "  └── figures/           # Paper figures (t-SNE, etc.)"
echo ""
echo "Key output files:"
echo "  results/multiseed/multiseed_summary.json    # All probe results (mean+std)"
echo "  results/dann/*/dann_results.json            # DANN results per model"
echo "  results/ensemble_ablation/ensemble_ablation.json"
echo "  results/scaling_curve/scaling_curve.json"
echo ""
