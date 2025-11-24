#!/usr/bin/env python3
import argparse
import os
import json
import re
import logging
import concurrent.futures
import subprocess
from pathlib import Path
import random

from langchain_ollama import OllamaLLM as Ollama

# ── Paths (adjust if needed) ────────────────────────────────────────────────────
REPORT_DIR = ".../embed/per_patient_json"  # The directory where the json reports are
OUTPUT_DIR = Path("result")  # the directory in which you want to save outputs

# ── Prompt ──────────────────────────────────────────────────────────────────────
PROMPT = r"""
You are a board-certified breast radiologist and a biostatistician specializing in longitudinal breast cancer risk modeling. You are given, for ONE patient at a time, a sequence of mammogram exams (oldest to newest) and their corresponding clinical history, all in a single JSON object.


ASSUME:
- The exams array is ordered from oldest to newest.
- One or more views or clinical fields may be missing or null.
- You will also receive the corresponding images referenced in "images" for each exam.

YOUR TASKS (reason step by step, but ONLY output JSON):

1. For each exam (internally):
   - Carefully inspect the mammogram images and the clinical information.
   - Infer the overall level of suspicion: completely normal, probably benign, suspicious, or highly suspicious.
   - Consider changes over time: new lesions, growth, new calcifications, progression of asymmetry, or new architectural distortion.

2. Use the FULL longitudinal trajectory (all exams up to the most recent) to estimate cancer risk from NOW (the time of the last exam) over the next 8 years.

   Define risk_y as:
   - risk_y = P(developing breast cancer within y years from now),
   - where y = 1, 2, ..., 8,
   - expressed as a probability between 0 and 1 (inclusive).

   These should be interpreted as CUMULATIVE risks, so they MUST satisfy:
   - 0 < risk_1y < risk_2y < ... < risk_8y < 1.

   Heuristics you MUST follow (do NOT ignore these):
   - If all exams are consistently regular or benign (BIRADS 1 and 2 conceptually), and there are no strong clinical risk factors, keep all risks low and slowly increasing (e.g., in the range of 0.00 to 0.10 by 8 years).
   - If there are benign but stable findings over many years (BIRADS 3 conceptually) without progression, keep risks modest and gently rising.
   - If the LATEST exam shows a NEW or CLEARLY PROGRESSIVE suspicious lesion (BIRADS 4 to 5 conceptually), increase near-term risks (1 to 3 years) and also the later horizons.
   - If clinical factors indicate substantial background risk (e.g., BRCA carrier, strong family history, or prior cancer), shift the overall risk curve upward across all horizons.

   IMPORTANT:
   - You are NOT a calibrated medical risk model. These outputs are approximate, for research use only.
   - Still, you MUST obey probability logic and monotonicity as stated above.

3. Estimate a TIME-TO-EVENT summary from the longitudinal pattern.

   Define:
   - time_to_event.expected_years: a positive real number representing your best estimate of when cancer would most likely develop, IF it produces, measured in years from now.
   - time_to_event.bucket: one of the following discrete categories:
       "< 1 year",
       "1 to 3 years",
       "3 to 5 years",
       "5 to 8 years",
       "> 8 years_or_uncertain"
   - time to event reasoning: a SHORT textual explanation (2 to 4 sentences) tying your estimate to the imaging trajectory and clinical risk (new irregular mass on most recent exam after years of benign imaging suggests elevated near-term risk).

   Heuristics:
   - If the latest exam is essentially regular with long-standing benign imaging, choose a longer bucket (e.g., "5 to 8 years" or "> 8 years_or_uncertain").
   - If the latest exam is newly suspicious or highly suspicious, choose a shorter bucket (e.g., "<1 year" or "1 to 3 years").
   - Ensure consistency: if you choose a very short time to event bucket, early-horizon risks (risk_1y, risk_2y) should be meaningfully higher than later ones.

4. OUTPUT FORMAT (VERY IMPORTANT):

Return ONLY a single JSON object with NO extra commentary, in the following format:

{
  "patient_id": "<copy from input>",
  "risk_estimates": {
    "risk_1y": <float in [0,1]>,
    "risk_2y": <float in [0,1]>,
    "risk_3y": <float in [0,1]>,
    "risk_4y": <float in [0,1]>,
    "risk_5y": <float in [0,1]>,
    "risk_6y": <float in [0,1]>,
    "risk_7y": <float in [0,1]>,
    "risk_8y": <float in [0,1]>
  },
  "time_to_event": {
    "expected_years": <float > 0>,
    "bucket": "<one of: \"< 1 year\", \"1 to 3 years\", \"3 to 5 years\", \"5 to 8 years\", \"> 8 years_or_uncertain\">"
  }
}

CONSTRAINTS:
- Do NOT hallucinate impossible probabilities (all must be between 0 and 1).
- Enforce monotonicity: risk_1y < risk_2y < ... < risk_8y.
- Do NOT output any treatment recommendations or clinical management advice.
- Do NOT output anything outside of the JSON object.
""".strip()

# ── Risk keys for sanity checks ────────────────────────────────────────────────
RISK_KEYS = [f"risk_{i}y" for i in range(1, 9)]

# ── Helpers ────────────────────────────────────────────────────────────────────

def ensure_ollama_model(model_name: str):
    """
    Ensure the Ollama model is available locally.
    If not, attempt to `ollama pull <model_name>`.
    """
    try:
        show = subprocess.run(
            ["ollama", "show", model_name],
            capture_output=True,
            text=True,
        )
        if show.returncode != 0:
            print(f"📥 Model '{model_name}' not found locally. Pulling with `ollama pull {model_name}`...")
            pull = subprocess.run(["ollama", "pull", model_name])
            if pull.returncode != 0:
                raise RuntimeError(f"Failed to pull Ollama model '{model_name}'.")
        else:
            print(f"✅ Ollama model '{model_name}' is available.")
    except FileNotFoundError:
        raise RuntimeError("`ollama` CLI not found. Please install Ollama and ensure it is on your PATH.")


def list_json_reports(root: str):
    all_files = sorted([
        fn for fn in os.listdir(root)
        if fn.lower().endswith(".json")
    ])
    return all_files


def call_ollama(
    model: str,
    system_prompt: str,
    patient_json_str: str,
    timeout_seconds: int = 600,
    few_shot_jsons=None,
):
    """
    Call Ollama with a timeout, passing:
    - the system prompt
    - two example patient JSON objects (input-only few-shot)
    - the TARGET patient JSON object
    Returns the raw text response or None on timeout.
    """
    if few_shot_jsons is None:
        few_shot_jsons = []

    def _query():
        query = system_prompt

        if few_shot_jsons:
            query += (
                "\n\nBelow are two EXAMPLE patient JSON objects. "
                "They are provided ONLY to show the structure of the input. "
                "Do NOT output answers for these examples.\n"
            )
            for i, ex_json in enumerate(few_shot_jsons, start=1):
                query += f"\n\nExample patient {i} JSON (INPUT ONLY):\n{ex_json}\n"

        query += (
            "\n\nNow here is the JSON object for the TARGET patient that you must analyze. "
            "Follow ALL the instructions above and output ONLY the requested JSON object for this TARGET patient.\n\n"
            f"{patient_json_str}"
        )

        ollama = Ollama(model=model, temperature=0)
        logging.getLogger().setLevel(logging.ERROR)
        return ollama.invoke(query)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_query)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return None


def extract_json_block(text: str) -> str:
    """
    Extract the first {...} block from the text and clean trailing commas.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return ""
    s = text[start : end + 1]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def enforce_risk_monotonicity(risks: dict, eps: float = 1e-3) -> dict:
    """
    Make sure:
      0 <= risk_1y < risk_2y < ... < risk_8y <= 1
    Fill missing keys by small increments.
    """
    cleaned = {}
    prev = 0.0
    n = len(RISK_KEYS)
    for i, key in enumerate(RISK_KEYS):
        raw = risks.get(key, None)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = prev + eps

        val = max(0.0, min(1.0, val))

        if val <= prev + eps:
            val = prev + eps

        remaining = n - i - 1
        if remaining > 0:
            upper_bound = 1.0 - remaining * eps
            if val > upper_bound:
                val = upper_bound
        else:
            val = min(val, 1.0 - eps)

        cleaned[key] = float(val)
        prev = val
    return cleaned


def sanitize_time_to_event(tte: dict) -> dict:
    """
    Ensure expected_years > 0 and bucket is valid.
    """
    allowed = {
        "< 1 year",
        "1 to 3 years",
        "3 to 5 years",
        "5 to 8 years",
        "> 8 years_or_uncertain",
    }
    expected = tte.get("expected_years", 5.0)
    try:
        expected = float(expected)
    except (TypeError, ValueError):
        expected = 5.0
    if expected <= 0:
        expected = 5.0

    bucket = tte.get("bucket", "> 8 years_or_uncertain")
    if bucket not in allowed:
        if expected < 1:
            bucket = "< 1 year"
        elif expected < 3:
            bucket = "1 to 3 years"
        elif expected < 5:
            bucket = "3 to 5 years"
        elif expected < 8:
            bucket = "5 to 8 years"
        else:
            bucket = "> 8 years_or_uncertain"

    return {
        "expected_years": float(expected),
        "bucket": bucket,
    }


def normalize_result(text: str, fallback_patient_id: str):
    """
    Parse the model output into the required JSON structure:
    {
      "patient_id": "...",
      "risk_estimates": {...},
      "time_to_event": {...}
    }
    """
    if not text or not text.strip():
        return {
            "patient_id": fallback_patient_id,
            "risk_estimates": {k: 0.0 for k in RISK_KEYS},
            "time_to_event": {
                "expected_years": 5.0,
                "bucket": "> 8 years_or_uncertain",
            },
        }, "", False

    block = extract_json_block(text)
    if not block:
        return {
            "patient_id": fallback_patient_id,
            "risk_estimates": {k: 0.0 for k in RISK_KEYS},
            "time_to_event": {
                "expected_years": 5.0,
                "bucket": "> 8 years_or_uncertain",
            },
        }, "", False

    try:
        data = json.loads(block)
    except Exception:
        return {
            "patient_id": fallback_patient_id,
            "risk_estimates": {k: 0.0 for k in RISK_KEYS},
            "time_to_event": {
                "expected_years": 5.0,
                "bucket": "> 8 years_or_uncertain",
            },
        }, block, False

    patient_id = data.get("patient_id", fallback_patient_id)

    risk_dict = data.get("risk_estimates", {}) or {}
    risk_clean = enforce_risk_monotonicity(risk_dict)

    tte_dict = data.get("time_to_event", {}) or {}
    tte_clean = sanitize_time_to_event(tte_dict)

    out = {
        "patient_id": patient_id,
        "risk_estimates": risk_clean,
        "time_to_event": tte_clean,
    }
    return out, block, True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Longitudinal breast cancer risk estimation via Ollama LLM from JSON reports (with dynamic few-shot)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name (anas/video-llava:test, qwen2.5vl, blaifa/InternVL3, etc.)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=-1,
        help="Number of JSON reports to process (-1 for all).",
    )
    parser.add_argument(
        "--report_dir",
        default=REPORT_DIR,
        help="Directory containing patient JSON reports.",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Base directory to save risk JSON outputs.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-patient timeout in seconds (default 600).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )

    ensure_ollama_model(args.model)

    # Make sure output_dir is a Path and build a model-specific subdir
    base_output_dir = Path(args.output_dir)
    model_sanitized = args.model.replace(":", "_").replace("/", "_")
    output_dir = base_output_dir / model_sanitized / "fewshot"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_json_reports(args.report_dir)
    if args.num != -1:
        files = files[: args.num]

    processed_count = 0
    skipped_timeout = 0
    failed_parse = 0

    # Precompute full paths for convenience
    file_paths = [os.path.join(args.report_dir, f) for f in files]

    for idx, (fname, report_path) in enumerate(zip(files, file_paths), start=1):
        # Load TARGET patient JSON
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                patient_obj = json.load(f)
        except Exception as e:
            print(f"⚠️  Skipping unreadable JSON {fname}: {e}")
            continue

        patient_id = patient_obj.get("patient_id", os.path.splitext(fname)[0])
        patient_json_str = json.dumps(patient_obj, ensure_ascii=False, indent=2)

        # ----- Dynamic few-shot: pick 2 random OTHER JSONs as examples -----
        # Candidate indices exclude the current index
        if len(files) > 1:
            # Build a list of other indices
            other_indices = [i for i in range(len(files)) if i != (idx - 1)]
            # Sample up to 2 others
            sample_count = min(2, len(other_indices))
            chosen_indices = random.sample(other_indices, sample_count)
            few_shot_jsons = []
            for j in chosen_indices:
                ex_path = file_paths[j]
                try:
                    with open(ex_path, "r", encoding="utf-8") as ef:
                        ex_obj = json.load(ef)
                    few_shot_jsons.append(json.dumps(ex_obj, ensure_ascii=False, indent=2))
                except Exception:
                    # If any example fails to load, just skip it
                    continue
        else:
            few_shot_jsons = []

        # Call Ollama with dynamic few-shot
        try:
            raw_text = call_ollama(
                model=args.model,
                system_prompt=PROMPT,
                patient_json_str=patient_json_str,
                timeout_seconds=args.timeout,
                few_shot_jsons=few_shot_jsons,
            )
            if raw_text is None:
                print(f"⏱️ Timeout (> {args.timeout}s) for {fname}. Saving fallback.")
                skipped_timeout += 1
                result = {
                    "patient_id": patient_id,
                    "risk_estimates": {k: 0.0 for k in RISK_KEYS},
                    "time_to_event": {
                        "expected_years": 5.0,
                        "bucket": "> 8 years_or_uncertain",
                    },
                }
            else:
                result, cleaned_block, ok = normalize_result(raw_text, fallback_patient_id=patient_id)
                if not ok:
                    failed_parse += 1

        except Exception as e:
            print(f"❌  Ollama call failed on {fname}: {e}")
            result = {
                "patient_id": patient_id,
                "risk_estimates": {k: 0.0 for k in RISK_KEYS},
                "time_to_event": {
                    "expected_years": 5.0,
                    "bucket": "> 8 years_or_uncertain",
                },
            }
            failed_parse += 1

        # Save result
        base = os.path.splitext(fname)[0]
        out_path = output_dir / f"{base}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        processed_count += 1
        if processed_count % 10 == 0:
            print(f"✅ Processed {processed_count} patients")

    print("────────────────────────────────────────────")
    print(f"✅ Total patients processed: {processed_count}")
    print(f"⏭️  Skipped due to timeout: {skipped_timeout}")
    print(f"🧹 JSON parse fallbacks: {failed_parse}")


if __name__ == "__main__":
    main()
