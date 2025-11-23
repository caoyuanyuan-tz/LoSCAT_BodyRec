# LoSCAT Body-Region Classification (ViT, 12-Class) — Evaluation

Pretrained **Vision Transformer** (ViT) for 12-class body-region recognition with **side-merged** labels:

`SCFA, NECK, CHEST, ABDM, UBCK, LBCK, ARM, FRM, HND, THI, LEG, FT`

This repo lets you **run inference** and **evaluate** on *your own data*.
No training code is included.

## Install
```bash
pip install -r requirements.txt

## Single-image inference

python scripts/inference_predict.py \
  --image example_data/example_img_01.jpg \
  --hf_repo YOUR_HF_USERNAME/loscat-vit-12cls \
  --hf_file vit/best_vit_12cls.pt \
  --json_out outputs_example/pred_example.json

## Batch inference(folder)

python scripts/batch_inference.py \
  --folder /path/to/images \
  --hf_repo YOUR_HF_USERNAME/loscat-vit-12cls \
  --hf_file vit/best_vit_12cls.pt \
  --out_csv outputs_example/predictions.csv

## Evaluate predictions

