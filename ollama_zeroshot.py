#!/usr/bin/env python3
import argparse
import os
import json
import re
import requests
from PIL import Image
from langchain_ollama import OllamaLLM as Ollama
import logging
import concurrent.futures

# ── Paths ────────────────────────────────────────────────────────────────────────
# IMAGE_DIR  = "/mnt/data1/raiyan/breast_cancer/datasets/vindr/merged_images"
# REPORT_DIR = "/mnt/data1/raiyan/breast_cancer/datasets/vindr/merged_reports"   # unused but kept for parity
# OUTPUT_DIR = "/mnt/data1/raiyan/breast_cancer/MedGemma/result"  # adjust if desired
IMAGE_DIR="/mnt/data1/raiyan/breast_cancer/MedGemma/data2/images"
REPORT_DIR="/mnt/data1/raiyan/breast_cancer/MedGemma/data3/reports"
OUTPUT_DIR= "/mnt/data1/raiyan/breast_cancer/MedGemma/result"

# IMAGE_DIR="/mnt/data1/raiyan/breast_cancer/datasets/dmid/png_images/all_images"
# REPORT_DIR="/mnt/data1/raiyan/breast_cancer/datasets/dmid/Documents/GROUND-TRUTH-REPORTS"
# OUTPUT_DIR     = "/mnt/data1/raiyan/breast_cancer/MedGemma/result"

# ── Prompts ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a board certified breast radiologist with lots of experience 
in interpreting screening and diagnostic mammograms. You are meticulous, 
up to date with the latest BIRADS guidelines, and always provide clear, concise, 
and clinically actionable reports.
""".strip()

USER_PROMPT = """
I am providing you with a mammogram image. The image has all 4 breast views of a patient shown together. 
The upper two views are the craniocaudal (CC) views of each breast, right and left,
and the lower two views are the mediolateral oblique (MLO) views of each breast, right and left.
Your task is to analyze the image and provide a structured report in JSON format.

For analyzing the image, at first glance, you should identify the breast density using the ACR classification,
which includes:
- ACR A: Almost entirely fatty
- ACR B: Scattered fibroglandular densities
- ACR C: Heterogeneously dense
- ACR D: Extremely dense

Then, you should summarize any findings and abnormalities, from the images. Mention in which view the findings
are present and say "Healthy Breast. No Findings" for normal breasts. Findings include
Mass, Suspicious Calcification, Architectural Distortion, Asymmetry, Focal Asymmetry, Global Asymmetry,
Suspicious Lymph Nodes, Nipple Retraction, Skin Retraction, Skin Thickening. 
There may be multiple findings in a single image.

Finally, assign a BIRADS category based on the findings:
- BIRADS 1: Negative (no abnormalities)
- BIRADS 2: Benign (no suspicion of cancer)
- BIRADS 3: Probably benign (short-term follow-up recommended)
- BIRADS 4: Suspicious abnormality (biopsy needed)
- BIRADS 5: Highly suggestive of malignancy (high probability of cancer)


Finally, tell if the image is healthy (yes) or not (no) in status.
                
Here is the JSON format you should follow for your response:
{
    "breast_density": "<ACR A|B|C|D> followed by a brief description of the density",
    "findings": "<Summary of any abnormalities as described above in one sentence>",
    "birads": "<1|2|3|4|5> followed by a brief description of the BIRADS category>",
    "status": "<yes|no>"
}

Here is the image you need to analyze: 

""".strip()


# ── Helpers ──────────────────────────────────────────────────────────────────────
def clean_json(text: str) -> str:
    """
    Extracts the first {…} block from text, drops anything before/after,
    and removes any trailing commas before closing braces/brackets.
    """
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return text  # nothing to extract
    s = text[start : end + 1]
    s = re.sub(r",\s*([}\]])", r"\1", s)  # remove trailing commas
    return s

def normalize_result(text: str):
    """
    Fix common formatting issues in model outputs:
    - Remove escaped underscores (\\_ → _)
    - Ensure 'birads' value is always a string
    - Fallback to "N/A" if parsing fails
    """
    # 1. Remove escaped underscores
    text = text.replace("\\_", "_")

    # 2. Extract JSON block (as you already had)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"breast_density": "N/A", "findings": "N/A", "birads": "N/A"}, text, False
    s = text[start:end+1]

    # 3. Remove trailing commas
    s = re.sub(r",\s*([}\]])", r"\1", s)

    try:
        data = json.loads(s)
        # Fix birads → always string
        if "birads" in data:
            data["birads"] = str(data["birads"])
        return {
            "breast_density": data.get("breast_density", "N/A"),
            "findings": data.get("findings", "N/A"),
            "birads": data.get("birads", "N/A"),
            "status": data.get("status", "N/A")
        }, s, True
    except Exception:
        return {"breast_density": "N/A", "findings": "N/A", "birads": "N/A", "status":"N/A"}, s, False


def list_images(root: str):
    all_files = sorted([
        fn for fn in os.listdir(root)
        if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ])
    return all_files

def call_ollama(model: str, system_prompt: str, user_prompt: str, image_path: str, timeout_seconds: int = 600):
    """
    Call Ollama with a timeout. If it takes longer than timeout_seconds, return None.
    """

    def _query():
        query = system_prompt + "\n" + user_prompt + "\n" + image_path
        ollama = Ollama(model=model, temperature=0)
        logging.getLogger().setLevel(logging.ERROR)  # Suppress INFO logs
        return ollama.invoke(query)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_query)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            print(f"⏱️ Timeout (> {timeout_seconds}s) for {image_path}. Skipping.")
            return None

def main():
    parser = argparse.ArgumentParser(description="Generate mammogram reports via Ollama VLMs with timeout and JSON cleaning.")
    parser.add_argument("--model", required=True, help="Ollama model name (e.g., qwen2.5-vl, llava:latest, etc.)")
    parser.add_argument("--num", type=int, default=-1, help="Number of images to process (-1 for all)")
    parser.add_argument("--image_dir", default=IMAGE_DIR, help="Directory of images")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Directory to save JSON reports")
    parser.add_argument("--timeout", type=int, default=600, help="Per-image timeout in seconds (default 600 = 10 minutes)")
    args = parser.parse_args()
    
    # ── Dynamic output directory ────────────────────────────────────────────────
    model_sanitized = args.model.replace(":", "_").replace("/", "_")
    output_dir = f"/mnt/data1/raiyan/breast_cancer/MedGemma/result/aug3_{model_sanitized}/zeroshot"
    os.makedirs(output_dir, exist_ok=True)

    # os.makedirs(args.output_dir, exist_ok=True)

    files = list_images(args.image_dir)
    if args.num != -1:
        files = files[:args.num]

    processed_count = 0
    skipped_timeout = 0
    failed_parse = 0

    for idx, fname in enumerate(files, start=1):
        img_path = os.path.join(args.image_dir, fname)

        # Quick sanity check the image is openable (also avoids sending broken paths)
        try:
            _ = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️  Skipping unreadable image {fname}: {e}")
            continue

        try:
            raw_text = call_ollama(
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=USER_PROMPT,
                image_path=img_path,
                timeout_seconds=args.timeout
            )
            if raw_text is None:
                result = {"image_id": img_path, "breast_density": "N/A", "findings": "N/A", "birads": "N/A", "status": "N/A"}
        # except requests.exceptions.Timeout:
        #     print(f"⏱️  Timeout (> {args.timeout}s) on {fname}. Skipping.")
        #     skipped_timeout += 1
        #     continue
        except Exception as e:
            print(f"❌  Ollama call failed on {fname}: {e}")
            # Save N/A in this case too
            result = {"image_id": img_path, "breast_density": "N/A", "findings": "N/A", "birads": "N/A", "status": "N/A"}
            base = os.path.splitext(fname)[0]
            out_path = os.path.join(args.output_dir, f"{base}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            failed_parse += 1
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"✅ Processed {processed_count} reports")
            continue

        # Clean & parse (fallback to N/A if needed)
        fields, cleaned_or_raw, ok = normalize_result(raw_text)
        if not ok:
            failed_parse += 1

        # Save result
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{base}.json")
        annotated = {"image_id": img_path, **fields}
        # Optionally keep the raw model output to help debug parsing
        # annotated["_raw"] = cleaned_or_raw

        with open(out_path, "w") as f:
            json.dump(annotated, f, indent=2)

        processed_count += 1
        if processed_count % 10 == 0:
            print(f"✅ Processed {processed_count} reports")

    print("────────────────────────────────────────────")
    print(f"✅ Total reports processed: {processed_count}")
    print(f"⏭️  Skipped due to timeout: {skipped_timeout}")
    print(f"🧹 JSON parse fallbacks (saved as N/A): {failed_parse}")

if __name__ == "__main__":
    main()
