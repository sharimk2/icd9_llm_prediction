# SDoH ICD-9 Code Detection from Clinical Notes

Admission-level, multi-label prediction of Social Determinants of Health (SDoH) **ICD-9 V-codes** from **MIMIC-III** clinical notes using reasoning-capable LLMs.

---

## Overview

Using **GPT-5-mini**, this pipeline achieves:

| Metric                |   Score   |
| :-------------------- | :-------: |
| Exact Match (Amended) | **72.2%** |
| Macro F1              | **66.5%** |
| Micro F1              | **89.1%** |

These results show that LLMs can reliably convert unstructured EHR narratives into structured SDoH billing codes.

---

## Data Access (Important)

* This project uses **MIMIC-III v1.4** clinical notes.
* **Dataset is NOT included** in this repository.
* Access requires **PhysioNet credentialing** and completion of required training. Please obtain the data directly from PhysioNet and set local paths accordingly.

---

## Repository Contents

* `admission_level_evaluation_per_note.py` — Main evaluation script for admission-level predictions
* `sdoh_icd9_prompt.txt` — Optimized reasoning prompt template
* `evaluation_metrics.json` — Full experimental metrics

---

## Requirements

```bash
pip install pandas numpy scikit-learn openai
```

---

## Usage

Set your OpenAI API key and run the evaluator:

```bash
export OPENAI_API_KEY='your-api-key-here'
python admission_level_evaluation_per_note.py
```

Outputs are written to `evaluation_metrics.json`.

---

## SDoH V-Codes Detected

|  ICD-9 | Domain              | Description                                        |
| :----: | :------------------ | :------------------------------------------------- |
|  V60.0 | Housing & Resources | Homelessness                                       |
|  V60.2 | Housing & Resources | Financial hardship / inadequate material resources |
|  V60.4 | Social Support      | No family caregiver                                |
|  V62.0 | Employment          | Unemployment                                       |
|  V62.5 | Legal               | Legal problems                                     |
| V15.41 | Abuse History       | Physical/sexual abuse                              |
| V15.42 | Abuse History       | Emotional abuse                                    |
| V61.41 | Family History      | Family alcoholism                                  |

---

## Citation

If you use this repository, please cite:

```
@inproceedings{khanLandesSdohIcd9Llm2025,
  title = {Social Determinants of Health Prediction for {{ICD-9}} Code with Reasoning Models},
  booktitle = {Proceedings of the 5th Machine Learning for Health Symposium},
  author = {Khan, Sharim and Landes, Paul and Sun, Jimeng},
  date = {2025-12},
  location = {San Diego, USA},
  abstract = {Social Determinants of Health correlate with patient outcomes but are rarely captured in structured data. Recent attention has been given to automatically extracting these markers from clinical text to supplement diagnostic systems with knowledge of patients’ social circumstances. Large language models demonstrate strong performance in identifying Social Determinants of Health labels from sentences. However, prediction in large admissions or longitudinal notes is challenging given long distance dependencies. In this paper, we explore hospital admission multi-label Social Determinants of Health ICD-9 code classification on the MIMIC-III dataset using reasoning models and traditional large language models. We exploit existing ICD-9 codes for prediction on admissions, which achieved a 89\% F1. Our contributions include our findings, missing SDoH codes in 139 admissions, and code to reproduce the results.},
  file = {/Users/landes/opt/var/zotero/storage/L97NID2P/Khan-and-Landes-SdohIcd9Llm.pdf}
}

```
