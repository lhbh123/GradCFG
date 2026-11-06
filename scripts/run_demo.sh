#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
MODEL_PATH="${MODEL_PATH:-sd_model/tiny-sd}"
DATA_DIR="${DATA_DIR:-data}"
LABEL="${LABEL:-backpack}"
RUN_ID="${RUN_ID:-demo_${LABEL}}"
ATTACK="${ATTACK:-attack}"
BATCH_SIZE="${BATCH_SIZE:-5}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-200}"
ATTACK_EPOCHS="${ATTACK_EPOCHS:-4001}"
RUN_INV_REV="${RUN_INV_REV:-0}"
case "$ATTACK" in attack|attack-lowmem) ;; *) echo 'ATTACK must be attack or attack-lowmem' >&2; exit 1;; esac
if (( TRAIN_EPOCHS < 1 || ATTACK_EPOCHS < 1 || BATCH_SIZE < 1 )); then exit 1; fi
TRAIN_DIR="ckpts/model_ckpt/${RUN_ID}/${LABEL}"
ATTACK_DIR="outputs/${RUN_ID}/attack"
if [[ -e "ckpts/model_ckpt/${RUN_ID}" || -e "outputs/${RUN_ID}" ]]; then
  echo 'Run already exists. Choose a new RUN_ID.' >&2; exit 1
fi
python run.py train --config configs/train.json --model_name "$MODEL_PATH" \
  --data_dir "$DATA_DIR" --prompt "$LABEL" --output_dir "ckpts/model_ckpt/${RUN_ID}" \
  --epochs "$TRAIN_EPOCHS" --batch_size "$BATCH_SIZE" --num_samples "$BATCH_SIZE"
python run.py "$ATTACK" --config configs/gradcfg.json --model_name "$RUN_ID" \
  --model_path "$MODEL_PATH" --label_name "$LABEL" --batch_size "$BATCH_SIZE" \
  --num_epochs "$ATTACK_EPOCHS" --output_dir "$ATTACK_DIR"
if [[ "$RUN_INV_REV" == 1 ]]; then
  FINAL=$((ATTACK_EPOCHS - 1))
  python run.py sample --model "$MODEL_PATH" --prompt "$LABEL" \
    --latents "${ATTACK_DIR}/dummy_images_epoch${FINAL}.pth" \
    --text "${ATTACK_DIR}/text_emb_epoch${FINAL}.pth" \
    --original-checkpoint "${TRAIN_DIR}/epoch_0.pt" \
    --finetuned-checkpoint "${TRAIN_DIR}/epoch_${TRAIN_EPOCHS}.pt" \
    --output "outputs/${RUN_ID}/inv_rev.png"
fi
