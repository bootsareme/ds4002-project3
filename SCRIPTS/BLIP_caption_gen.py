import os
import re
import torch
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from difflib import SequenceMatcher

DATA_DIR = "DATA"
EXCEL_PATH = os.path.join(DATA_DIR, "meme compilation.xlsx")
OUTPUT_PATH = "OUTPUT/results_blip.csv"
TOP_K = 5 


def clean_text(text):
    text = str(text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9.,!?\'\" ]+", "", text)
    return text.lower().strip()


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def score_caption(candidate, references):
    candidate = clean_text(candidate)
    scores = [similarity(candidate, clean_text(ref)) for ref in references]
    return max(scores) if scores else 0


def load_model():
    device = "cpu"

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    model.eval()
    return processor, model, device


def generate_candidates(image, processor, model, device, n=10):
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            top_p=0.9,
            temperature=1.6,   # slightly higher for more variation
            num_return_sequences=n
        )

    captions = [
        processor.decode(o, skip_special_tokens=True)
        for o in outputs
    ]

    return captions


def main():
    print("[INFO] Humor-oriented BLIP → CSV pipeline")

    processor, model, device = load_model()
    df = pd.read_excel(EXCEL_PATH)

    # This will store: {column_name: [top captions]}
    final_table = {}

    for column in df.columns:
        image_path = os.path.join(DATA_DIR, f"{column}.png")

        if not os.path.exists(image_path):
            print(f"[WARNING] Missing image: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")

        gt_captions = [
            str(c) for c in df[column].dropna()
            if str(c).strip()
        ]

        # Generate candidates
        candidates = generate_candidates(image, processor, model, device, n=10)

        # Score candidates
        scored = []
        for c in candidates:
            s = score_caption(c, gt_captions)
            scored.append((c, s))

        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top K
        top_captions = [c for c, _ in scored[:TOP_K]]

        print("=" * 60)
        print(f"IMAGE: {column}.png")
        for i, (c, s) in enumerate(scored[:TOP_K]):
            print(f"[{i+1}] ({s:.3f}) {c}")

        final_table[column] = top_captions

    # Convert to DataFrame (columns = memes, rows = captions)
    output_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in final_table.items()]))
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[INFO] Saved CSV to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()