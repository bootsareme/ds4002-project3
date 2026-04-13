import os
import re
import torch
import pandas as pd
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from difflib import SequenceMatcher

DATA_DIR = "DATA"
EXCEL_PATH = os.path.join(DATA_DIR, "meme compilation.xlsx")
OUTPUT_PATH = "OUTPUT/results_blipv2.csv"
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-flan-t5-xl"
    )

    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16
    ).to(device)

    model.eval()
    return processor, model, device


def generate_candidates(image, processor, model, device, n=15):
    prompt = "Write a short funny meme caption."

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(device, torch.float16)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=1.3,
            top_p=0.95,
            num_return_sequences=n,
        )

    captions = [
        processor.decode(o, skip_special_tokens=True)
        for o in outputs
    ]

    return list(set(captions))  # remove duplicates


def main():
    print("[INFO] BLIP-2 humor pipeline (GPU)")

    processor, model, device = load_model()
    df = pd.read_excel(EXCEL_PATH)

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

        # Generate
        candidates = generate_candidates(image, processor, model, device, n=15)

        # Score
        scored = [(c, score_caption(c, gt_captions)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Top K
        top_captions = [c for c, _ in scored[:TOP_K]]

        print("=" * 60)
        print(f"IMAGE: {column}.png")
        for i, (c, s) in enumerate(scored[:TOP_K]):
            print(f"[{i+1}] ({s:.3f}) {c}")

        final_table[column] = top_captions

    # Convert to DataFrame
    output_df = pd.DataFrame(dict([
        (k, pd.Series(v)) for k, v in final_table.items()
    ]))

    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n[INFO] Saved CSV to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
