"""
Export trained PyTorch model to ONNX format.
ONNX is ~10x lighter than PyTorch and works on Streamlit Cloud.

Usage:
    python export_to_onnx.py
    python export_to_onnx.py --model output/ffb_model_best.pth --out ffb_model.onnx
"""

import argparse
import torch
import torch.nn as nn
import torchvision.models as models

CLASS_NAMES = ["ripe", "underripe", "unripe", "rotten", "empty"]


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_feat, num_classes),
    )
    return model


def export(model_path: str, onnx_path: str):
    model = build_model(len(CLASS_NAMES))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"✅ Exported to {onnx_path}")
    print("   Copy ffb_model.onnx to your Streamlit repo root and redeploy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="output/ffb_model_best.pth")
    parser.add_argument("--out",   default="ffb_model.onnx")
    args = parser.parse_args()
    export(args.model, args.out)
