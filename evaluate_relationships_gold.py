import pandas as pd
import sys
from sklearn.metrics import precision_score, recall_score, f1_score

# --- Helper: Normalize column names and string values ---
def normalize(df):
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        'metric a': 'metric_a',
        'metric_b': 'metric_b',
        'relationship type': 'relationship_type',
        'relationship': 'relationship_type'  # fallback
    })
    return df.apply(lambda x: x.str.strip().str.upper() if x.dtype == 'object' else x)

# --- Load CSV Files ---
try:
    gold_df = pd.read_csv("data/gold_telecom_metric_relationships.csv")
    pred_df = pd.read_csv("data/test_telecom_metric_relationships.csv")
except Exception as e:
    print(f"❌ File load error: {e}")
    sys.exit(1)

# --- Normalize Columns ---
gold_df = normalize(gold_df)
pred_df = normalize(pred_df)

# --- Column Validation ---
required_cols = {'metric_a', 'relationship_type', 'metric_b'}
if not required_cols.issubset(set(gold_df.columns)):
    print(f"❌ Missing columns in gold file: {required_cols - set(gold_df.columns)}")
    sys.exit(1)
if not required_cols.issubset(set(pred_df.columns)):
    print(f"❌ Missing columns in predicted file: {required_cols - set(pred_df.columns)}")
    sys.exit(1)

# --- Create Sets of (metric_a, relationship_type, metric_b) Tuples ---
gold_set = set(tuple(x) for x in gold_df[['metric_a', 'relationship_type', 'metric_b']].values)
pred_set = set(tuple(x) for x in pred_df[['metric_a', 'relationship_type', 'metric_b']].values)

# --- Evaluation Metrics ---
true_positives = len(gold_set & pred_set)
false_positives = len(pred_set - gold_set)
false_negatives = len(gold_set - pred_set)

precision = (true_positives / (true_positives + false_positives) if pred_set else 0)*100
recall = (true_positives / (true_positives + false_negatives) if gold_set else 0)*100
f1 = 2 * (precision * recall) / (precision + recall) if precision + recall else 0

# --- Print Report ---
print("\n📊 Relationship Evaluation Report")
print("--------------------------------------")
print(f"✅ Total Gold Relationships      : {len(gold_set)}")
print(f"✅ Total Predicted Relationships : {len(pred_set)}")
print(f"🎯 True Positives                : {true_positives}")
print(f"❌ False Positives               : {false_positives}")
print(f"⚠️  False Negatives               : {false_negatives}")
print("--------------------------------------")
print(f"🎯 Precision                     : {precision:.2f}")
print(f"🎯 Recall                        : {recall:.2f}")
print(f"🎯 F1 Score                      : {f1:.2f}")

# --- Optional: Save Mismatches to CSV ---
false_pos = pred_df[pred_df.apply(lambda row: (row['metric_a'], row['relationship_type'], row['metric_b']) in (pred_set - gold_set), axis=1)]
false_neg = gold_df[gold_df.apply(lambda row: (row['metric_a'], row['relationship_type'], row['metric_b']) in (gold_set - pred_set), axis=1)]

false_pos.to_csv("data/false_positives.csv", index=False)
false_neg.to_csv("data/false_negatives.csv", index=False)

print("\n💾 False positives saved to: data/false_positives.csv")
print("💾 False negatives saved to: data/false_negatives.csv")
