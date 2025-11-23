import argparse, os, csv
from pathlib import Path
from scripts.inference_predict import predict

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def iter_images(folder):
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            yield str(p)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder containing images (recursive).")
    ap.add_argument("--hf_repo", required=True)
    ap.add_argument("--hf_file", required=True)
    ap.add_argument("--model", default="vit_base_patch16_384")
    ap.add_argument("--out_csv", default="predictions.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    rows = []
    for img_path in iter_images(args.folder):
        try:
            r = predict(img_path, args.hf_repo, args.hf_file, model_name=args.model, topk=5)
            rows.append([img_path, r["pred_label"], f"{r['confidence']:.6f}"])
        except Exception as e:
            rows.append([img_path, "ERROR", f"{e}"])
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "pred_label", "confidence"])
        w.writerows(rows)
    print(f"[Saved] {args.out_csv} ({len(rows)} rows)")
