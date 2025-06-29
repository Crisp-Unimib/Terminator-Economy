import pandas as pd
import re
import gc
import os
import torch
import time
import random
import numpy as np
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

# ----------------------------- SETUP -----------------------------
def set_seed(seed: int = 28):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in .env file")

# ----------------------------- PROMPT -----------------------------
def build_prompt(row, model_name):
    examples = {
        "orca_mini": [
            "Profession: Architect\nTask: Designing a sustainable building.\nExample Evaluation: [4, \"Robotics can assist in model construction, while Image Processing Systems evaluate the designs against environmental standards. LLMs could aid in researching sustainable materials and methods, though the creative and integrative aspects of design might not fully leverage AI capabilities.\"]"
        ],
        "mistral": [
            "Profession: Architect\nTask: Designing a sustainable building.\nExample Evaluation: [4, \"Robotics can assist in model construction, while Image Processing Systems evaluate the designs against environmental standards. LLMs could aid in researching sustainable materials and methods, though the creative and integrative aspects of design might not fully leverage AI capabilities.\"]"
        ],
        "openchat": [
            "Profession: Architect\nTask: Designing a sustainable building.\nExample Evaluation: [4, \"Robotics can assist in model construction, while Image Processing Systems evaluate the designs against environmental standards. LLMs could aid in researching sustainable materials and methods, though the creative and integrative aspects of design might not fully leverage AI capabilities.\"]",
            "Profession: Medical Researcher\nTask: Analyzing genetic data to predict disease risk.\nExample Evaluation: [5, \"Image Processing Systems can analyze genetic patterns, LLMs can process vast amounts of research to support findings, and Robotics can automate the handling and preparation of genetic samples, collectively enhancing the accuracy and efficiency of disease prediction.\"]",
            "Profession: Urban Planner\nTask: Creating a city's traffic flow optimization plan.\nExample Evaluation: [5, \"Image Processing Systems analyze current traffic patterns and congestion points, while LLMs can process and incorporate relevant research and regulations. Robotics could be used for the physical implementation of traffic control devices. Their combined use allows for a comprehensive and efficient optimization plan.\"]",
            "Profession: Environmental Scientist\nTask: Monitoring and analyzing deforestation rates.\nExample Evaluation: [4, \"Image Processing Systems can provide accurate, real-time analysis of satellite imagery to track deforestation. LLMs could assist in correlating deforestation rates with climate data and policies, offering insights into trends and causes. Robotics, though less directly involved, could aid in physical data collection on the ground.\"]",
            "Profession: Financial Analyst\nTask: Predicting stock market trends.\nExample Evaluation: [3, \"LLMs can analyze news articles and financial reports to gauge market sentiment, but their predictions may lack precision without quantitative analysis. Image Processing Systems and Robotics have limited applicability in directly predicting stock market trends, highlighting the need for specialized AI in finance.\"]"
        ]
    }

    prefix = """
Assess the capability of AI technologies—Large Language Models (LLMs), Image Processing Systems, and Robotics—either individually or in combination, to perform specific tasks within various professions.

For each task, consider:
- Whether LLMs, known for their advanced text understanding and generation, can contribute to the task's completion.
- If Image Processing Systems, with their ability to analyze and interpret visual data, are applicable.
- The role of Robotics in executing tasks that require physical action or manipulation.
- The potential for these AI models to complement each other, enhancing the overall effectiveness.

For each evaluation:
- Rate the combined or individual capability of LLMs, Image Processing Systems, and Robotics on a scale from 1 to 5.
- Provide a detailed justification in the format: [rating, "justification"].
"""

    shot_key = "openchat" if "openchat" in model_name else ("mistral" if "mistral" in model_name else "orca_mini")
    examples_text = "\n\n".join(examples[shot_key])
    task = f"Given the profession: {row['Title']}, and a specific task: '{row['Task']}', evaluate the combined or individual capability of these AI technologies to perform the task."
    return f"{prefix}\n\n{examples_text}\n\n{task}"

# ----------------------------- INFERENCE + CHECKPOINT -----------------------------
def evaluate_with_model(model_name, model_path, df):
    checkpoint_path = f"{model_name}_partial.csv"
    if os.path.exists(checkpoint_path):
        print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
        df_partial = pd.read_csv(checkpoint_path)
        df = df.merge(df_partial[["Task ID", f"{model_name}_ratings", f"{model_name}_motivation"]],
                      on="Task ID", how="left", suffixes=("", "_ckpt"))
    else:
        df[f"{model_name}_ratings"] = None
        df[f"{model_name}_motivation"] = None

    to_process = df[df[f"{model_name}_ratings"].isna()].copy()
    if to_process.empty:
        print(f"✅ All rows already evaluated for {model_name}")
        return df

    print(f"🔍 Loading model: {model_name}")
    llm = LLM(model=model_path, dtype="half", trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=2048)

    prompts = [build_prompt(row, model_name) for _, row in to_process.iterrows()]
    outputs = llm.generate(prompts, sampling_params)

    ratings, justifications = [], []
    for output in tqdm(outputs, desc=f"Inference: {model_name}"):
        generated_text = output.outputs[0].text
        match = re.search(r"\[.*?\]", generated_text)
        if match:
            try:
                parsed = eval(match.group(0))
                ratings.append(int(parsed[0]))
                justifications.append(str(parsed[1]))
            except:
                ratings.append(None)
                justifications.append(generated_text.strip())
        else:
            ratings.append(None)
            justifications.append(generated_text.strip())

    to_process[f"{model_name}_ratings"] = ratings
    to_process[f"{model_name}_motivation"] = justifications
    df.update(to_process)

    df.to_csv(checkpoint_path, index=False, encoding="utf-8")
    print(f"💾 Partial results saved to {checkpoint_path}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()
    return df

# ----------------------------- AGGREGATION -----------------------------
def mode(row, colnames):
    values = [v for v in [row[c] for c in colnames] if isinstance(v, int)]
    if not values:
        return None
    return min(values) if len(set(values)) == 3 else pd.Series(values).mode().min()

# ----------------------------- SUMMARIZATION -----------------------------
def generate_summary(row):
    user_message = f"""You are a text summarization model.
        you will receive in input 3 differents text dealing with LLMs, image processing systems and robotics capabilities. 
        Text 1: [{row['motivation_mistral']}] 
        Text 2: [{row['motivation_orca_mini']}]
        Text 3: [{row['open_chat_motivation']}]
        You must summarize those 3 text in a unique one. 
        You must provide the summary into square brackets.
        You must return only the summary.
        You cannot return other elements.
        """

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen/qwen-2.5-72b-instruct",
                "messages": [{"role": "user", "content": user_message}],
                "top_p": 1,
                "temperature": 0
            }
        )
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            match = re.findall(r'\[(.*?)\]', content)
            return match[0] if match else content.strip()
        else:
            return None
    except Exception as e:
        return None

# ----------------------------- MAIN -----------------------------
def main():
    set_seed(28)
    df = pd.read_excel("Task Statements.xlsx", engine="openpyxl")

    if "Task ID" not in df.columns:
        raise KeyError("🛑 'Task ID' column not found in the input Excel file.")

    models = [
        ("orca_mini", "TheBloke/orca_mini_v3_7B-GPTQ"),
        ("mistral", "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"),
        ("openchat", "TheBloke/openchat-3.5-0106-GPTQ")
    ]

    for model_name, model_path in models:
        df = evaluate_with_model(model_name, model_path, df)

    rating_cols = [f"{name}_ratings" for name, _ in models]
    df["TEAI_rating"] = df.apply(lambda row: mode(row, rating_cols), axis=1)

    summary_path = "TEAI_result.xlsx"
    if os.path.exists(summary_path):
        print("🔁 Resuming summarization from:", summary_path)
        df_existing = pd.read_excel(summary_path, engine="openpyxl")
        if "teai_summary" in df_existing.columns:
            df = df.merge(df_existing[["Task ID", "teai_summary"]], on="Task ID", how="left")

    failures = []

    for task_id, row in tqdm(df[df["teai_summary"].isna()].set_index("Task ID").iterrows(), desc="Summarization"):
        summary = generate_summary(row)
        if summary:
            df.loc[df["Task ID"] == task_id, "teai_summary"] = summary
        else:
            failures.append(task_id)
        if len(failures) % 20 == 0:
            df.to_excel(summary_path, index=False, engine="openpyxl")

    df.to_excel(summary_path, index=False, engine="openpyxl")
    print("✅ All results saved to", summary_path)

    if failures:
        with open("teai_failed.log", "w") as f:
            for task_id in failures:
                f.write(f"{task_id}\n")
        print(f"⚠️ Logged {len(failures)} failed tasks in teai_failed.log")

if __name__ == "__main__":
    main()


