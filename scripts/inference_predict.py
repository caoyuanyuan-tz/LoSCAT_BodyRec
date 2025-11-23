import argparse, os, json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from huggingface_hub import hf_hub_download

from models.vit_12cls import build_vit, CLASSES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

def get_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def load_weights_from_hf(repo_id: str, filename: str):
    return hf_hub_download(repo_id=repo_id, filename=filename)

def predict(image_path: str, repo_id: str, weight_file: str, model_name="vit_base_patch16_384", topk=5, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vit(model_name=model_name, num_classes=len(CLASSES)).to(device)
    weights_path = load_weights_from_hf(repo_id, weight_file)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    x = get_transform()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
        conf, pred_idx = probs.max(dim=0)
        top_p, top_i = probs.topk(min(topk, len(CLASSES)))
    result = {
        "image_path": image_path,
        "pred_label": CLASSES[pred_idx.item()],
        "confidence": float(conf.item()),
        "topk": [{"label": CLASSES[i.item()], "prob": float(p.item())} for p, i in zip(top_p, top_i)]
    }
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to an image")
    p.add_argument("--hf_repo", required=True, help="HF repo id, e.g. yourname/loscat-vit-12cls")
    p.add_argument("--hf_file", required=True, help="Filename inside HF repo, e.g. vit/best_vit_12cls.pt")
    p.add_argument("--model", default="vit_base_patch16_384")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--json_out", default="")
    args = p.parse_args()

    res = predict(args.image, args.hf_repo, args.hf_file, model_name=args.model, topk=args.topk)
    print(f"Predicted: {res['pred_label']}  (confidence={res['confidence']:.3f})")
    print("Top-k:", ", ".join([f"{d['label']}({d['prob']:.2f})" for d in res["topk"]]))
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[Saved] {args.json_out}")
