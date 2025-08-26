"""
SDOH ICD-9 Code Detection from Clinical Notes

This script evaluates GPT-4o-mini's performance on detecting Social Determinants 
of Health (SDOH) ICD-9 V-codes from MIMIC-III clinical notes using a per-note 
processing approach with admission-level aggregation.

Author: [Your name here]
Date: 2024
"""

import pandas as pd
import numpy as np
from openai import OpenAI
import os
import time
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, hamming_loss, precision_recall_fscore_support
)
from sklearn.preprocessing import MultiLabelBinarizer

# Initialize OpenAI client
if 'OPENAI_API_KEY' not in os.environ:
    raise EnvironmentError("Please set OPENAI_API_KEY environment variable")

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Load dataset
print("Loading SDOH dataset...")
df = pd.read_csv('sdoh_icd9_dataset.csv')
print(f"Loaded {len(df)} notes from {df['HADM_ID'].nunique()} admissions")

with open('sdoh_icd9_prompt.txt', 'r') as f:
    prompt_template = f.read()

# Define target SDOH codes
TARGET_CODES = ['V600', 'V602', 'V604', 'V620', 'V625', 'V1541', 'V1542', 'V1584', 'V6141']

def parse_codes(codes_str):
    """Parse codes from various string formats"""
    if pd.isna(codes_str) or str(codes_str).strip() == '':
        return set()
    
    codes_str = str(codes_str).replace('[', '').replace(']', '').replace('"', '').replace("'", "")
    if ',' in codes_str:
        codes = [c.strip() for c in codes_str.split(',')]
    elif ';' in codes_str:
        codes = [c.strip() for c in codes_str.split(';')]
    else:
        codes = [codes_str.strip()]
    
    return set(c for c in codes if c.upper() != 'NONE' and c.strip() != '')

def call_openai_api(text, max_chars=100000):
    """Call OpenAI API with system/user message separation"""
    # Truncate if too long
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Note truncated due to length...]"
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": f"Analyze this clinical note and identify SDOH codes:\n\n{text}"}
                ],
                max_tokens=100,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return "ERROR"

def parse_llm_response(response):
    """Parse LLM response to extract codes from triple backticks"""
    if response == "ERROR" or not response:
        return set()
    
    # Extract content between triple backticks
    import re
    backtick_pattern = r'```(.*?)```'
    matches = re.findall(backtick_pattern, response, re.DOTALL)
    
    if matches:
        response = matches[0].strip()
    else:
        # Fallback: use the whole response
        response = response.strip()
    
    # Handle "None" response
    if response.lower().strip() == "none":
        return set()
    
    # Extract codes from response
    # Remove common prefixes/suffixes
    response = response.replace("Answer:", "").replace("Codes:", "").strip()
    
    # Split by common delimiters
    codes = []
    for delimiter in [',', ';', ' ', '\n']:
        if delimiter in response:
            codes = [c.strip() for c in response.split(delimiter)]
            break
    else:
        codes = [response.strip()]
    
    # Filter valid codes
    valid_codes = set()
    for code in codes:
        code = code.strip().upper()
        if code in TARGET_CODES:
            valid_codes.add(code)
    
    return valid_codes

def process_admission_per_note(admission_df):
    """Process each note individually and aggregate results"""
    all_predicted_codes = set()
    note_results = []
    
    # Sort by chart date to maintain chronological order
    admission_df = admission_df.copy()
    admission_df['CHARTDATE'] = pd.to_datetime(admission_df['CHARTDATE'])
    admission_df = admission_df.sort_values('CHARTDATE')
    
    for _, row in admission_df.iterrows():
        category = row['NOTE_CATEGORY']
        date = row['CHARTDATE'].strftime('%Y-%m-%d') if pd.notna(row['CHARTDATE']) else 'Unknown'
        text = str(row['FULL_TEXT']).strip()
        
        print(f"    Processing {category} note ({date})...")
        
        # Call LLM for this individual note
        llm_response = call_openai_api(text)
        predicted_codes = parse_llm_response(llm_response)
        
        print(f"      Predicted: {sorted(predicted_codes) if predicted_codes else 'None'}")
        
        # Add to aggregate
        all_predicted_codes.update(predicted_codes)
        
        note_results.append({
            'category': category,
            'date': date,
            'predicted_codes': list(predicted_codes),  # Convert set to list for JSON
            'llm_response': llm_response
        })
        
        # Small delay to avoid rate limiting
        time.sleep(0.2)
    
    return all_predicted_codes, note_results

def calculate_multilabel_metrics(y_true, y_pred, code_labels):
    """Calculate comprehensive multi-label classification metrics using precision_recall_fscore_support"""
    
    print("\n=== MULTI-LABEL CLASSIFICATION METRICS ===")
    
    # Overall metrics
    print(f"\n1. OVERALL METRICS:")
    print(f"   Subset Accuracy (Exact Match): {accuracy_score(y_true, y_pred):.3f}")
    print(f"   Hamming Loss: {hamming_loss(y_true, y_pred):.3f}")
    
    # Use precision_recall_fscore_support for comprehensive metrics
    # Micro averages
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    # Macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    print(f"\n2. MICRO AVERAGES (global performance):")
    print(f"   Precision: {micro_precision:.3f}")
    print(f"   Recall: {micro_recall:.3f}")
    print(f"   F1-Score: {micro_f1:.3f}")
    
    print(f"\n3. MACRO AVERAGES (per-class performance):")
    print(f"   Precision: {macro_precision:.3f}")
    print(f"   Recall: {macro_recall:.3f}")
    print(f"   F1-Score: {macro_f1:.3f}")
    
    print(f"\n4. WEIGHTED AVERAGES (support-weighted):")
    print(f"   Precision: {weighted_precision:.3f}")
    print(f"   Recall: {weighted_recall:.3f}")
    print(f"   F1-Score: {weighted_f1:.3f}")
    
    # Per-class metrics
    print(f"\n5. PER-CLASS METRICS:")
    for i, code in enumerate(code_labels):
        true_count = np.sum(y_true[:, i])
        pred_count = np.sum(y_pred[:, i])
        support = int(class_support[i])
        print(f"   {code}: P={class_precision[i]:.3f}, R={class_recall[i]:.3f}, F1={class_f1[i]:.3f} (Support:{support}, True:{true_count}, Pred:{pred_count})")
    
    # Label statistics
    print(f"\n6. LABEL STATISTICS:")
    print(f"   Admissions with no labels: {np.sum(np.sum(y_true, axis=1) == 0)}")
    print(f"   Admissions with 1+ labels: {np.sum(np.sum(y_true, axis=1) > 0)}")
    print(f"   Average labels per admission: {np.mean(np.sum(y_true, axis=1)):.2f}")
    print(f"   Max labels per admission: {np.max(np.sum(y_true, axis=1))}")
    
    # Return comprehensive metrics
    return {
        'subset_accuracy': accuracy_score(y_true, y_pred),
        'hamming_loss': hamming_loss(y_true, y_pred),
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_metrics': {
            code: {
                'precision': float(class_precision[i]),
                'recall': float(class_recall[i]),
                'f1': float(class_f1[i]),
                'support': int(class_support[i])
            }
            for i, code in enumerate(code_labels)
        }
    }

# Process all admissions
print(f"\nProcessing {df['HADM_ID'].nunique()} admissions...")

results = []
admission_counter = 0

for hadm_id, admission_group in df.groupby('HADM_ID'):
    admission_counter += 1
    if admission_counter % 10 == 0:
        print(f"\nProcessing admission {admission_counter}...")
    
    # Get true labels (from first row since all rows have same admission-level labels)
    first_row = admission_group.iloc[0]
    true_codes = parse_codes(first_row['ADMISSION_TRUE_CODES'])
    manual_codes = parse_codes(first_row['ADMISSION_MANUAL_LABELS'])
    
    # Process each note individually
    predicted_codes, note_results = process_admission_per_note(admission_group)
    
    # Store results
    results.append({
        'HADM_ID': hadm_id,
        'SUBJECT_ID': first_row['SUBJECT_ID'],
        'IS_GAP_CASE': first_row['IS_GAP_CASE'],
        'NUM_NOTES': len(admission_group),
        'NOTE_CATEGORIES': ','.join(admission_group['NOTE_CATEGORY'].tolist()),
        'TEXT_LENGTH': sum(len(str(row['FULL_TEXT'])) for _, row in admission_group.iterrows()),
        'TRUE_CODES': ','.join(sorted(true_codes)) if true_codes else '',
        'MANUAL_CODES': ','.join(sorted(manual_codes)) if manual_codes else '',
        'PREDICTED_CODES': ','.join(sorted(predicted_codes)) if predicted_codes else '',
        'NOTE_RESULTS': json.dumps(note_results),
        'TRUE_CODES_SET': true_codes,
        'MANUAL_CODES_SET': manual_codes,
        'PREDICTED_CODES_SET': predicted_codes
    })

# Create results DataFrame
results_df = pd.DataFrame(results)

print(f"\n=== EVALUATION COMPLETE ===")
print(f"Processed {len(results_df)} admissions")
print(f"API errors: {sum(1 for r in results if 'ERROR' in str(r.get('NOTE_RESULTS', '')))}") 

# Calculate metrics using manual labels as ground truth
print(f"\n=== EVALUATING AGAINST MANUAL LABELS ===")

# Prepare data for multi-label metrics
mlb = MultiLabelBinarizer(classes=TARGET_CODES)
y_true = mlb.fit_transform([r['MANUAL_CODES_SET'] for r in results])
y_pred = mlb.transform([r['PREDICTED_CODES_SET'] for r in results])

metrics = calculate_multilabel_metrics(y_true, y_pred, TARGET_CODES)

# Also calculate metrics against true codes for comparison
print(f"\n=== EVALUATING AGAINST TRUE ICD-9 CODES ===")
y_true_icd = mlb.transform([r['TRUE_CODES_SET'] for r in results])
metrics_icd = calculate_multilabel_metrics(y_true_icd, y_pred, TARGET_CODES)

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f'admission_level_results_per_note_{timestamp}.csv'
results_df.to_csv(results_file, index=False)

# Save metrics summary
metrics_summary = {
    'evaluation_timestamp': timestamp,
    'processing_method': 'per_note',
    'total_admissions': len(results_df),
    'api_errors': sum(1 for r in results if 'ERROR' in str(r.get('NOTE_RESULTS', ''))),
    'manual_labels_metrics': metrics,
    'true_codes_metrics': metrics_icd
}

with open(f'admission_level_metrics_per_note_{timestamp}.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print(f"\n=== FILES SAVED ===")
print(f"Results: {results_file}")
print(f"Metrics: admission_level_metrics_per_note_{timestamp}.json")

print(f"\n=== SUMMARY ===")
print(f"Best performing metric vs manual labels: Micro F1 = {metrics['micro_f1']:.3f}")
print(f"Exact match accuracy: {metrics['subset_accuracy']:.3f}")
print(f"Average prediction errors per admission: {metrics['hamming_loss']:.3f}")