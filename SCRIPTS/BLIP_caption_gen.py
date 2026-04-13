import os
import json
import torch
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

DATA_DIR = "../DATA"
EXCEL_PATH = os.path.join(DATA_DIR, "meme compilation.xlsx")
OUTPUT_PATH = "../OUTPUT/results.json"


def load_model():
    device = 'cpu'
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    model.eval()  # important for inference
    return processor, model, device


def generate_caption(image, processor, model, device):
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=3   # smaller beams = faster on CPU
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def main():
    print("[INFO] Running BLIP on CPU")

    processor, model, device = load_model()

    df = pd.read_excel(EXCEL_PATH)

    results = []

    for column in df.columns:
        image_path = os.path.join(DATA_DIR, f"{column}.png")

        if not os.path.exists(image_path):
            print(f"[WARNING] Missing image: {image_path}")
            continue

        # Clean ground truth captions
        gt_captions = [
            str(c).strip()
            for c in df[column].dropna()
            if str(c).strip()
        ]

        image = Image.open(image_path).convert("RGB")

        # Generate caption
        generated = generate_caption(image, processor, model, device)

        # Print to stdout
        print("=" * 60)
        print(f"IMAGE: {column}.png")
        print(f"GENERATED: {generated}")

        print("GROUND TRUTH:")
        for c in gt_captions:
            print(f" - {c}")

        # Store results
        results.append({
            "image": column,
            "generated": generated,
            "ground_truth": gt_captions
        })

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[INFO] Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
