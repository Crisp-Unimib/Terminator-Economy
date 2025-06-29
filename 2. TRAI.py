import os
import re
import ast
import time
import json
import nltk
import requests
import threading
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor, as_completed
from Levenshtein import distance as levenshtein_distance
import argparse


# ----------------------------- SETUP -----------------------------
load_dotenv()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found. Please set it in the .env file.")

# Rate limiting
RATE_LIMIT_PER_MINUTE = 60
REQUESTS_PER_SECOND = RATE_LIMIT_PER_MINUTE / 60
lock = threading.Lock()
last_request_time = [0]

def rate_limited_request():
    with lock:
        elapsed = time.time() - last_request_time[0]
        sleep_time = max(0, 1 / REQUESTS_PER_SECOND - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_request_time[0] = time.time()

# ----------------------------- UTILS -----------------------------
def preprocess_text(text, stop_words):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join([w for w in text.split() if w not in stop_words])

def compare_strings(str1, str2, max_distance=15):
    return levenshtein_distance(str1, str2) <= max_distance

def extract_json(response_text):
    try:
        match = re.search(r"\{.*?\}", response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None
    except json.JSONDecodeError:
        return None

def process_row(row):
    rate_limited_request()

    task_id = row["Task ID"]
    title = row["Title"]
    task = row["Task"]
    summary = row["teai_summary"]


    # Prompt rinforzato per assicurare solo JSON
    user_message = f"""You are an expert on the impact of artificial intelligence on the labour market. 
You will receive as input the title of a job profession, a task for this job profession and a description of the impact of different 
Artificial Intelligence technologies on the task provided as input.
You must return a numerical value from 1 to 5 that measures the level of engagement of 
the artificial intelligence in the execution of the task provided as input, based on the description provided as input.
A value of 1 equals ‘no engagement’, while a value of 5 equals ‘replacement of the human in the execution of the task by artificial intelligence’.
If a task can be performed by artificial intelligence and requires only complementarity by the human, it must be considered fully automated with a rate of 5.
In this perspective, you must return a binary flag of 1 if a significant human complementarity is required and 0 if it is not required.

Here are the elements: 
Job Title: [{title}]
Job task: [{task}]
Description of the impact of AI on the task: [{summary}]

Let's think step by step. 
Return ONLY a JSON object with the following keys: "job_title", "job_task", "ai_engagement_level", "flag", "reasoning".
No additional text, no explanations outside the JSON.
"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "qwen/qwen-2.5-72b-instruct",
                "messages": [{"role": "user", "content": user_message}],
                "top_p": 1,
                "temperature": 0
            },
            timeout=30
        )

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            response_data = extract_json(content)
            if not response_data:
                return None

            title_in = preprocess_text(title, stop_words)
            task_in = preprocess_text(task, stop_words)
            title_out = preprocess_text(response_data.get("job_title", ""), stop_words)
            task_out = preprocess_text(response_data.get("job_task", ""), stop_words)

            if compare_strings(title_in, title_out) and compare_strings(task_in, task_out):
                return {
                    "ai_engagement_level": response_data.get("ai_engagement_level"),
                    "flag": response_data.get("flag"),
                    "reasoning": response_data.get("reasoning")
                }
        return None
    except Exception:
        return None

# ----------------------------- MAIN -----------------------------
def main(input_path, output_path):
    set_seed(28)
    df = pd.read_excel(input_path, engine="openpyxl")

    if "ai_engagement_level" in df.columns:
        df_filtered = df[df["ai_engagement_level"].isna()]
    else:
        df["ai_engagement_level"] = None
        df["ai_engagement_reasoning"] = None
        df["flag_engagement"] = None
        df_filtered = df

    print(f"🧠 Processing {len(df_filtered)} rows...")

    results = {}
    failures = []

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {
            executor.submit(process_row, row): idx
            for idx, row in df_filtered.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="TRAI Evaluation"):
            idx = futures[future]
            task_id = df_filtered.iloc[idx]["Task ID"]
            try:
                result = future.result()
                if result:
                    results[task_id] = result
                else:
                    failures.append(task_id)
            except Exception:
                failures.append(task_id)

    for task_id, result in results.items():
        df.loc[df["Task ID"] == task_id, "ai_engagement_level"] = result["ai_engagement_level"]
        df.loc[df["Task ID"] == task_id, "ai_engagement_reasoning"] = result["reasoning"]
        df.loc[df["Task ID"] == task_id, "flag_engagement"] = result["flag"]

    df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"✅ Results saved to: {output_path}")

    if failures:
        with open("trai_failed.log", "w") as f:
            for tid in failures:
                f.write(f"{tid}\n")
        print(f"⚠️ Logged {len(failures)} failures to trai_failed.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI engagement for each task.")
    parser.add_argument("--input", type=str, default="TEAI_result.xlsx", help="Input Excel file path")
    parser.add_argument("--output", type=str, default="TEAI_TRAI_final.xlsx", help="Output Excel file path")
    args = parser.parse_args()

    main(args.input, args.output)
