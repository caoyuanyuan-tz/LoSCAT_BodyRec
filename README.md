# LoSCAT Body-Region Classification (ViT, 12-Class) — Evaluation

Pretrained **Vision Transformer** (ViT) for 12-class body-region recognition with **side-merged** labels:

`SCFA, NECK, CHEST, ABDM, UBCK, LBCK, ARM, FRM, HND, THI, LEG, FT`

This repo lets you **run inference** and **evaluate** on *your own data*.
No training code is included.

## Install
```bash
pip install -r requirements.txt

```

## Single-image inference
```bash
python scripts/inference_predict.py \
  --image example_data/example_img_01.jpg \
  --hf_repo YOUR_HF_USERNAME/loscat-vit-12cls \
  --hf_file vit/best_vit_12cls.pt \
  --json_out outputs_example/pred_example.json
```
## Batch inference(folder)
```bash
python scripts/batch_inference.py \
  --folder /path/to/images \
  --hf_repo YOUR_HF_USERNAME/loscat-vit-12cls \
  --hf_file vit/best_vit_12cls.pt \
  --out_csv outputs_example/predictions.csv
```
## Evaluate predictions
Prepare a ground-truth CSV:
```pgsql
image_path,region
/path/to/img1.jpg,HND
/path/to/img2.jpg,SCFA
```

```bash
python scripts/evaluate_predictions.py \
  --pred_csv outputs_example/predictions.csv \
  --gt_csv example_data/example_ground_truth.csv \
  --out_dir outputs_example

```
Outputs: classification_report.csv, raw/normalized confusion matrices, and a PNG heatmap.

## Weights (Hugging Face)
Weights are hosted on the Hugging Face Hub and downloaded automatically by the scripts.
Download Link
https://huggingface.co/cyycn20000509/loscat-body-region-classifier/blob/main/best_12cls_vit.pt
