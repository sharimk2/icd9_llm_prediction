# SDOH ICD-9 Code Detection from Clinical Notes

This repository contains the evaluation pipeline for detecting Social Determinants of Health (SDOH) ICD-9 V-codes from MIMIC-III clinical notes using GPT-4o-mini.

## Overview

The system achieves 69.1% subset accuracy on multi-label SDOH classification, with:
- Macro F1: 0.641
- Micro F1: 0.851
- Hamming Loss: 5.04%

## Files

1. **`admission_level_evaluation_per_note.py`** - Main evaluation script that processes clinical notes and predicts SDOH codes
2. **`sdoh_icd9_dataset.csv`** - Full curated dataset with 151 admissions
3. **`sdoh_icd9_prompt.txt`** - Optimized prompt template for GPT-4o-mini
4. **`evaluation_metrics.json`** - Complete evaluation metrics from our experiments

## Requirements

```bash
pip install pandas numpy scikit-learn openai
```

## Usage

Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

Run evaluation:
```bash
python admission_level_evaluation_per_note.py
```

## SDOH Categories Detected

- V600: Homelessness
- V602: Financial hardship
- V604: No social support
- V620: Unemployment
- V625: Legal problems
- V1541: Physical abuse history
- V1542: Emotional abuse history
- V6141: Family alcoholism
- V6142: Family substance abuse