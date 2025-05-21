```

                 _____                          _       _   _           
                |_   _|                        | |     | | | |          
                  | | _____      ____ _ _ __ __| |___  | |_| |__   ___  
                  | |/ _ \ \ /\ / / _` | '__/ _` / __| | __| '_ \ / _ \ 
                  | | (_) \ V  V / (_| | | | (_| \__ \ | |_| | | |  __/ 
                  \_/\___/ \_/\_/ \__,_|_|  \__,_|___/  \__|_| |_|\___| 
                                                                        
                                                                        
                  _____                   _             _               
                 |_   _|                 (_)           | |              
                   | | ___ _ __ _ __ ___  _ _ __   __ _| |_ ___  _ __   
                   | |/ _ \ '__| '_ ` _ \| | '_ \ / _` | __/ _ \| '__|  
                   | |  __/ |  | | | | | | | | | | (_| | || (_) | |     
                   \_/\___|_|  |_| |_| |_|_|_| |_|\__,_|\__\___/|_|     
                                                                        
                                                                        
                    _____                                               
                   |  ___|                                              
                   | |__  ___ ___  _ __   ___  _ __ ___  _   _          
                   |  __|/ __/ _ \| '_ \ / _ \| '_ ` _ \| | | |         
                   | |__| (_| (_) | | | | (_) | | | | | | |_| |         
                   \____/\___\___/|_| |_|\___/|_| |_| |_|\__, |         
                                                          __/ |         
                                                         |___/          


```



> 📝 This repository contains the official implementation of the paper  
> **"Towards the Terminator Economy: Assessing Job Exposure to AI through LLMs"**, accepted at **[IJCAI 2025](https://2025.ijcai.org/)** – _AI and Social Good Track_.


---

## 📦 Overview

**The Terminator Economy** is a two-stage AI evaluation pipeline that analyzes the capability of Large Language Models (LLMs), Image Processing Systems, and Robotics in performing specific occupational tasks.

The framework consists of:

1. **TEAI.py**: Evaluates tasks using multiple local LLMs and summarizes their outputs.
2. **TRAI.py**: Uses the generated summary to compute an AI engagement score and flags human complementarity.

Both stages are fully reproducible, checkpointable, and resume-safe.

---

## 🔐 API Requirement

This project requires an **API key from [OpenRouter.ai](https://openrouter.ai)** to access the summarization and engagement scoring models.

Please create a `.env` file in the root directory with the following content:

```env
API_KEY=your_openrouter_api_key_here
````

You can obtain a free key from OpenRouter after logging in.

---

## 🧠 TEAI.py — Task Evaluation with AI

⚠️ **Hardware Requirement:** This script requires access to a GPU. It loads and runs multiple open-source LLMs locally using the vLLM inference engine.

### 🔍 Input

* `Task Statements.xlsx` (must include the following columns):

  * `Task ID` (unique identifier)
  * `Title` (profession name)
  * `Task` (task description)

### ⚙️ What it does

* Loads 3 LLMs (e.g., Mistral, OpenChat, Orca) locally via vLLM.
* Constructs few-shot prompts and infers both a **rating** and a **motivation** for each task.
* Aggregates results using a conservative **mode function** (`TEAI_rating`).
* Summarizes all three justifications into one paragraph (`teai_summary`) using OpenRouter API.

### 💾 Output

* `TEAI_result.xlsx` including:

  * `Task ID`, `Title`, `Task`
  * `*_ratings`, `*_motivation`
  * `TEAI_rating`, `teai_summary`

### ▶️ Execution

```bash
python TEAI.py
```

Resumes automatically from checkpoints if interrupted.

---

## 🤖 TRAI.py — Task Rating with AI

### 🔍 Input

* `TEAI_result.xlsx` (output from previous step)

### ⚙️ What it does

* Sends `Title`, `Task`, and `teai_summary` to a 72B instruction-tuned model via API.
* Receives structured JSON containing:

  * `ai_engagement_level` (1–5 scale)
  * `flag` (0 = no human needed, 1 = human complementarity required)
  * `reasoning` (justification)
* Uses Levenshtein distance to validate output consistency.

### 💾 Output

* `TEAI_TRAI_final.xlsx` with added columns:

  * `ai_engagement_level`
  * `flag_engagement`
  * `ai_engagement_reasoning`
* `trai_failed.log` for failed or inconsistent rows

### ▶️ Execution

```bash
python TRAI.py --input TEAI_result.xlsx --output TEAI_TRAI_final.xlsx
```

---

## 📈 Output Format Summary

| Column                    | Description                                  |
| ------------------------- | -------------------------------------------- |
| `O*NET-SOC Code`          | Occupation Code	                             |
| `Task ID`                 | Unique identifier of the task                |
| `Title`, `Task`           | Occupation and associated task               |
| `*_ratings`               | Evaluation scores from each LLM              |
| `*_motivation`            | Model-specific reasoning                     |
| `TEAI_rating`             | TEAI rating                                  |
| `teai_summary`            | Unified summary of motivations               |
| `ai_engagement_level`     | TRAI rating                                  |
| `flag_engagement`         | Binary flag for human complementarity        |
| `ai_engagement_reasoning` | Justification for the final engagement score |
| `Date`                    | Date of task entry                           |
---
## 🧮 O*NET Weights for Index Aggregation

The provided `Task Ratings.xlsx` file includes, for each task, three key measures:

- **Frequency** (`Frequency of Task`): how often a task is performed within the occupation
- **Importance** (`Importance of Task`): how critical that task is for successful job performance
- **Aggregated relevance**: typically constructed by combining frequency and importance

These measures are used to **weight the TEAI and TRAI scores** in the aggregation phase (using the AIOE Toolkit), resulting in indices that are robust, interpretable, and representative of occupational reality.

### 📊 Dataset Structure

| Column              | Description                                                        |
|---------------------|--------------------------------------------------------------------|
| `O*NET-SOC Code`    | Standard occupational code (SOC, US taxonomy)                      |
| `Title`             | Occupation title                                                   |
| `Task ID`           | Unique task identifier                                             |
| `Task`              | Task description                                                   |
| `Scale Name`        | Metric type (`Frequency of Task`, `Importance of Task`, etc.)      |
| `Data Value`        | Estimated mean value (e.g., 4.5 out of 5)                          |
| `N`                 | Number of respondents who rated the task                           |
| `Standard Error`    | Standard error of the estimate                                     |
| `Lower CI Bound` / `Upper CI Bound` | 95% confidence interval bounds                     |
| `Date`              | Collection or update date (e.g., `08/2023`)                        |

Multiple rows for the same `Task ID` correspond to different metrics or scales (e.g., both frequency and importance).


## 🚀 Final Step — Compute the TEAI and TRAI Indexes

After running TEAI.py and TRAI.py, you can generate interpretable occupation-level automation indices using the official pipeline from:

👉 AIOE: AI Occupational Exposure Toolkit

This project provides a standardized and open-source framework to:

* 📊 Aggregate your task-level `TEAI_rating` and `ai_engagement_level` into occupation-level indices

* ⚖️ Apply employment-weighted averaging for robust labor market indicators

* 🌍 Enable cross-country and cross-sector automation comparisons

🔁 To ensure compatibility:

* Use `TEAI_result.xlsx` and `TEAI_TRAI_final.xlsx` as your base inputs

* Follow AIOE’s input formatting

* Run the aggregation pipeline to compute:

* The TEAI Index

* The TRAI Index

✅ Integrating with AIOE allows you to position your results within a broader ecosystem of empirical research on AI and the future of work.

## 🛡️ Reproducibility

* Seed fixed at `28` across all modules.
* Intermediate results stored to enable crash recovery.
* All prompts and model configurations are consistent and version-controlled.

---

## 📜 License

MIT License

---

## 🤝 Acknowledgments

* Qwen, OpenRouter.ai, vLLM contributors
* Mistral, OpenChat, and Orca model communities

---


