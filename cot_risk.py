#!/usr/bin/env python3
import argparse
import os
import json
import re
import logging
import concurrent.futures
import subprocess
from pathlib import Path

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

IMPORTANT: You'll need to think step-by-step internally, but your FINAL output must ONLY be the JSON object described below, with NO intermediate reasoning, NO explanations, and NO extra text.

-------------------------
INTERNAL REASONING STEPS
(Do NOT include these in your output)
-------------------------

1. Parse and understand the input
   - Read the patient-level JSON.
   - Note patient_id, demographics (age, family history, genetic risk, other risk factors).
   - Note the list of exams, their dates, ages, available views, and clinical notes (indication, symptoms, prior history, etc.).

2. Per-exam qualitative assessment
   For EACH exam in chronological order:
   - Inspect the mammogram images and the corresponding clinical information.
   - For that exam, internally determine:
     - Whether the imaging is entirely normal, probably benign, suspicious, or highly suspicious (conceptually similar to BIRADS 1 to 5).
     - Any notable findings or pattern ( masses, calcifications, asymmetry, architectural distortion, skin changes, suspicious lymph nodes).
   - Pay attention to image quality and density if visible (though you do not need to output them).
   - Keep this exam-level assessment in your internal reasoning; do NOT output it directly.

3. Longitudinal (temporal) reasoning
   - Compare exams over time:
     - Check for new lesions vs old lesions.
     - Assess whether existing lesions are stable, shrinking, or growing.
     - Look for new calcifications, evolving asymmetry, or new architectural distortion.
   - Internally classify the overall longitudinal pattern:
     - Consistently benign/normal with no worrisome change.
     - Probably benign but stable over many years.
     - New or progressive suspicious lesion on the latest exam after benign/normal history.
   - Integrate clinical risk:
     - Strong family history, BRCA status, prior cancer, or other risk factors should shift your internal notion of baseline risk upward.
   - Summarize (internally) a qualitative picture: ``overall low risk with stable benign findings'' vs ``elevated near-term risk due to new suspicious lesion''.

4. Construct the 1 to 8 year risk curve (discrete-time cumulative risk)
   - Define for each y in {1,2,3,4,5,6,7,8}:
     - risk_y = P(developing breast cancer within y years from now),
       expressed as a probability in [0,1].
   - These are CUMULATIVE risks from NOW (the time of the last exam), so they MUST satisfy:
     - 0 < risk_1y < risk_2y < ... < risk_8y < 1.
   - Use the following heuristics (do NOT ignore these):
     - If all exams are consistently regular or benign (BIRADS 1 to 2 conceptually) and there are no strong clinical risk factors:
       - Keep all risks low and slowly increasing (roughly in the range ~0.00 to 0.10 by 8 years).
     - If there are probably benign but stable findings over many years (BIRADS 3 conceptually) without progression:
       - Keep risks modest and gently rising ( moderate but smooth increase from year 1 to year 8).
     - If the LATEST exam shows a NEW or CLEARLY PROGRESSIVE suspicious lesion (BIRADS 4 to 5 conceptually):
       - Increase near-term risks (years 1 to 3) and also the later horizons, so that early risks are meaningfully higher than in benign cases.
     - If clinical factors indicate substantial background risk ( BRCA carrier, strong family history, prior cancer):
       - Shift the entire risk curve upward across all horizons while preserving monotonicity.
   - You are NOT a calibrated medical risk model. These outputs are approximate, for research use only.
   - After you draft the risk_1y to risk_8y values, internally check:
     - All are between 0 and 1 (inclusive).
     - risk_1y < risk_2y < ... < risk_8y.

5. Derive a time-to-event estimate
   - Based on your longitudinal assessment and risk curve, estimate:
     - time to event.expected years: a positive real number (float > 0) representing your best guess for when cancer would most likely develop, IF it produces, measured in years from now.
     - time_to_event.bucket: choose EXACTLY ONE of:
       - "<1 year"
       - "1 to 3 years"
       - "3 to 5 years"
       - "5 to 8 years"
       - ">=8 years_or_uncertain"
   - Heuristics:
     - If the latest exam is essentially regular with long-standing benign imaging and a low-risk curve:
       - Prefer longer buckets like "5 to 8 years" or ">=8 years or uncertain".
     - If the latest exam is newly suspicious or highly suspicious with elevated near-term risk:
       - Prefer shorter buckets like "<1 year" or "1 to 3 years".
     - Ensure basic consistency between the bucket and the risk curve:
       - If you choose a very short bucket ( "<1 year"), then risk_1y and early horizons should be noticeably higher relative to later benign scenarios.
   - Internally pick an expected_years value that is consistent with the bucket and the overall risk curve.

6. Final validation before output
   - Double-check that:
     - patient_id is copied exactly from the input.
     - All risk_y values are within [0,1].
     - risk_1y < risk_2y < ... < risk_8y.
     - time to event.expected years > 0.
     - time to event.bucket is one of the allowed strings.
   - Do NOT include any of your reasoning steps, explanations, or comments in the output.

-------------------------
FINAL OUTPUT FORMAT
(Output ONLY this JSON object)
-------------------------

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
    "bucket": "<one of: "<1 year", "1 to 3 years", "3 to 5 years", "5 to 8 years", ">=8 years or uncertain">"
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
        # Check if model exists
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


def call_ollama(model: str, system_prompt: str, patient_json_str: str, timeout_seconds: int = 600):
    """
    Call Ollama with a timeout, passing the system prompt plus the patient JSON.
    Returns the raw text response or None on timeout.
    """
    def _query():
        # We just concatenate system prompt + patient JSON in one message,
        # since langchain_ollama.OllamaLLM uses a single string interface.
        query = (
            system_prompt
            + "\n\nHere is the JSON object for ONE patient:\n\n"
            + patient_json_str
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
    s = re.sub(r",\s*([}\]])", r"\1", s)  # remove trailing commas
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

        # clamp
        val = max(0.0, min(1.0, val))

        # strictly increasing
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
        # pure fallback
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

    # patient_id
    patient_id = data.get("patient_id", fallback_patient_id)

    # risk_estimates
    risk_dict = data.get("risk_estimates", {}) or {}
    risk_clean = enforce_risk_monotonicity(risk_dict)

    # time_to_event
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
        description="Longitudinal breast cancer risk estimation via Ollama LLM from JSON reports."
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

    # Ensure model is present
    ensure_ollama_model(args.model)

    # Dynamic subdirectory per model
    model_sanitized = args.model.replace(":", "_").replace("/", "_")
    output_dir = os.path.join(args.output_dir, f"{model_sanitized}", "cot")
    os.makedirs(output_dir, exist_ok=True)

    files = list_json_reports(args.report_dir)
    if args.num != -1:
        files = files[: args.num]

    processed_count = 0
    skipped_timeout = 0
    failed_parse = 0

    for idx, fname in enumerate(files, start=1):
        report_path = os.path.join(args.report_dir, fname)

        # Load patient JSON
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                patient_obj = json.load(f)
        except Exception as e:
            print(f"⚠️  Skipping unreadable JSON {fname}: {e}")
            continue

        patient_id = patient_obj.get("patient_id", os.path.splitext(fname)[0])
        patient_json_str = json.dumps(patient_obj, ensure_ascii=False, indent=2)

        # Call Ollama
        try:
            raw_text = call_ollama(
                model=args.model,
                system_prompt=PROMPT,
                patient_json_str=patient_json_str,
                timeout_seconds=args.timeout,
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
        out_path = os.path.join(output_dir, f"{base}.json")
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
