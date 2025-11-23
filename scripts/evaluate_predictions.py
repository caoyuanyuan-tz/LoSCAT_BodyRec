import argparse, os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Keep the label order consistent with training
CLASSES = [
    'SCFA','NECK','CHEST','ABDM','UBCK','LBCK',
    'ARM','FRM','HND','THI','LEG','FT'
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="CSV from batch_inference.py (image_path,pred_label,confidence)")
    ap.add_argument("--gt_csv", required=True, help="Ground truth CSV with columns: image_path,region")
    ap.add_argument("--out_dir", default="eval_results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pred = pd.read_csv(args.pred_csv)
    gt   = pd.read_csv(args.gt_csv)

    # merge by image_path
    df = gt.merge(pred[["image_path", "pred_label"]], on="image_path", how="left")
    df = df.dropna(subset=["region", "pred_label"]).copy()

    y_true = df["region"].astype(str).tolist()
    y_pred = df["pred_label"].astype(str).tolist()

    # filter to known classes (optional)
    known = [y in CLASSES for y in y_true]
    df = df.loc[known]
    y_true = df["region"].tolist()
    y_pred = df["pred_label"].tolist()

    # metrics
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=CLASSES)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")

    # classification report
    rep = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES, digits=3, zero_division=0, output_dict=True)
    rep_df = pd.DataFrame(rep).transpose()
    rep_path = os.path.join(args.out_dir, "classification_report.csv")
    rep_df.to_csv(rep_path)
    print(f"[Saved] {rep_path}")

    # confusion matrices
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cm_df.to_csv(os.path.join(args.out_dir, "confusion_matrix_raw.csv"))

    # normalized (row-wise = per-class recall)
    cm_norm = cm.astype("float")
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums != 0)
    cmn_df = pd.DataFrame(cm_norm, index=CLASSES, columns=CLASSES)
    cmn_df.to_csv(os.path.join(args.out_dir, "confusion_matrix_norm.csv"))

    # heatmap
    plt.figure(figsize=(9,7))
    sns.heatmap(cmn_df, cmap="Blues", annot=True, fmt=".2f", vmin=0.0, vmax=1.0,
                cbar_kws={"label":"Per-class recall (row-normalized)"})
    plt.title("Normalized Confusion Matrix (ViT 12-class)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    out_png = os.path.join(args.out_dir, "confusion_matrix_norm.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[Saved] {out_png}")