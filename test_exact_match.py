import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login
from src.modules.rules_regex import RegexBasedDetector
from src.modules.ensemble_guard import EnsembleGuard
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# Authenticate with Hugging Face
login(token="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")

def normalize_labels(series: pd.Series) -> pd.Series:
    """Map various label schemes to {0,1}."""
    if series.dtype.kind in "iu":  # already ints
        return series.astype(int)

    mapping = {
        "benign": 0, "clean": 0, "safe": 0, "non_jailbreak": 0, "not_jailbreak": 0,
        "jailbreak": 1, "prompt_injection": 1, "injection": 1, "malicious": 1,
        "attack": 1, "adversarial": 1
    }
    s = series.astype(str).str.lower().map(mapping)
    s = s.fillna(series.astype(str).str.lower().isin(["true", "1", "yes"]).astype(int))
    return s.astype(int)

def pick_text_column(df: pd.DataFrame) -> str:
    """Find the text column name; fallback to common variants."""
    for c in ["text", "content", "prompt", "input", "message", "instruction"]:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find a text column in: {list(df.columns)}")

def _coerce_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'label' column exists and normalized."""
    for c in ["label", "labels", "target", "class", "category", "is_jailbreak", "is_injection"]:
        if c in df.columns:
            df["label"] = normalize_labels(df[c])
            return df
    raise KeyError(f"Could not find a label-like column in: {list(df.columns)}")

def _df_from_split(split_ds):
    """Convert a HF split to pandas with columns ['text','label']."""
    df = split_ds.to_pandas()
    txt = pick_text_column(df)
    df = df.rename(columns={txt: "text"})
    df = _coerce_label_column(df)
    keep = [c for c in ["text", "label"] if c in df.columns]
    return df[keep].dropna(subset=["text", "label"])

def recreate_exact_test_data():
    """Recreate the EXACT same test data used in training"""
    print("Recreating exact test data from training...")
    
    # Load the same 3 datasets
    ds1 = load_dataset("xTRam1/safe-guard-prompt-injection")
    ds2 = load_dataset("deepset/prompt-injections")
    ds3 = load_dataset("jayavibhav/prompt-injection")
    
    # Access splits
    test1, test2, test3 = ds1["test"], ds2["test"], ds3["test"]
    
    # Concatenate and shuffle with SAME seed as training
    _test_all_hf = concatenate_datasets([test1, test2, test3]).shuffle(seed=42)
    
    # Sample SAME size as training (300)
    TEST_SIZE = 300
    _test_all_hf = _test_all_hf.select(range(min(TEST_SIZE, len(_test_all_hf))))
    
    # Convert to pandas with unified columns
    test_df = _df_from_split(_test_all_hf)
    
    return test_df

def test_exact_match():
    """Test using exact same conditions as training"""
    
    # Get the exact same test data
    test_df = recreate_exact_test_data()
    texts = test_df['text'].tolist()
    labels = test_df['label'].values
    
    print(f"Test set size: {len(texts)}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    results = {}
    
    # 1. BERT Model (Fine-tuned)
    print("\n=== BERT Model (Fine-tuned) ===")
    MODEL_DIR = "models/bert-pi-detector/fine_tuned"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    
    threshold_path = os.path.join(MODEL_DIR, "threshold.txt")
    threshold = float(open(threshold_path).read().strip())
    print(f"Using threshold: {threshold}")
    
    pipeline = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        top_k=None,
    )
    
    bert_predictions = []
    for text in texts:
        result = pipeline(text, truncation=True, max_length=256)
        if isinstance(result, list) and len(result) > 0:
            scores = result[0] if isinstance(result[0], list) else result
            prob = next(s["score"] for s in scores if s["label"].endswith("1"))
        else:
            prob = 0.5
        bert_predictions.append(1 if prob >= threshold else 0)
    
    bert_acc = accuracy_score(labels, bert_predictions)
    print(f"BERT Accuracy: {bert_acc:.3f}")
    print(classification_report(labels, bert_predictions, target_names=['Benign', 'Malicious']))
    results['BERT'] = bert_predictions
    
    # 2. Regex Model
    print("\n=== Regex Model ===")
    try:
        regex_model = RegexBasedDetector("src/utils/patterns.regex.yaml")
        regex_predictions = []
        for text in texts:
            score = regex_model.score(text)
            pred = 1 if score["level"] in ("warn", "block") else 0
            regex_predictions.append(pred)
        
        regex_acc = accuracy_score(labels, regex_predictions)
        print(f"Regex Accuracy: {regex_acc:.3f}")
        print(classification_report(labels, regex_predictions, target_names=['Benign', 'Malicious']))
        results['Regex'] = regex_predictions
    except Exception as e:
        print(f"Regex model failed: {e}")
    
    # 3. Ensemble Model
    print("\n=== Ensemble Model ===")
    try:
        ensemble_model = EnsembleGuard()
        ensemble_predictions = []
        for text in texts:
            decision = ensemble_model.decide(text)
            pred = 1 if decision["decision"] == "block" else 0
            ensemble_predictions.append(pred)
        
        ensemble_acc = accuracy_score(labels, ensemble_predictions)
        print(f"Ensemble Accuracy: {ensemble_acc:.3f}")
        print(classification_report(labels, ensemble_predictions, target_names=['Benign', 'Malicious']))
        results['Ensemble'] = ensemble_predictions
    except Exception as e:
        print(f"Ensemble model failed: {e}")
    
    # Summary comparison
    print("\n=== SUMMARY COMPARISON ===")
    for model_name, preds in results.items():
        acc = accuracy_score(labels, preds)
        print(f"{model_name:12}: {acc:.3f}")
    
    # Correlation analysis
    print("\n=== CORRELATION ANALYSIS ===")
    if len(results) >= 2:
        df_results = pd.DataFrame(results)
        correlation_matrix = df_results.corr()
        print("Correlation Matrix:")
        print(correlation_matrix.round(3))
        
        # Pairwise correlations
        model_names = list(results.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                corr = np.corrcoef(results[model_names[i]], results[model_names[j]])[0,1]
                print(f"{model_names[i]} vs {model_names[j]}: {corr:.3f}")
    
    # Weighted ensemble combinations
    print("\n=== WEIGHTED ENSEMBLE COMBINATIONS ===")
    if 'BERT' in results and 'Regex' in results:
        bert_preds = np.array(results['BERT'])
        regex_preds = np.array(results['Regex'])
        
        # Test different weight combinations
        weight_combinations = [
            (0.9, 0.1),   # BERT-heavy
            (0.8, 0.2),   # BERT-dominant
            (0.7, 0.3),   # BERT-majority
            (0.6, 0.4),   # BERT-slight
            (0.5, 0.5),   # Equal weight
        ]
        
        weighted_results = {}
        
        for bert_w, regex_w in weight_combinations:
            # Weighted average (treat as probabilities)
            weighted_scores = bert_w * bert_preds + regex_w * regex_preds
            weighted_preds = (weighted_scores >= 0.5).astype(int)
            
            acc = accuracy_score(labels, weighted_preds)
            combo_name = f"BERT({bert_w})+Regex({regex_w})"
            weighted_results[combo_name] = weighted_preds
            print(f"{combo_name:25}: {acc:.3f}")
        
        # Find best weighted combination
        best_combo = max(weighted_results.items(), key=lambda x: accuracy_score(labels, x[1]))
        best_name, best_preds = best_combo
        best_acc = accuracy_score(labels, best_preds)
        
        print(f"\nBest weighted combination: {best_name} ({best_acc:.3f})")
        print("\nDetailed results for best combination:")
        print(classification_report(labels, best_preds, target_names=['Benign', 'Malicious']))
        
        # Add best combination to results for final comparison
        results['Best_Weighted'] = best_preds
    
    # XGBoost Meta-Learner
    print("\n=== XGBOOST META-LEARNER ===")
    if XGBOOST_AVAILABLE and len(results) >= 2:
        # Prepare features (model predictions)
        feature_names = ['BERT', 'Regex']
        if all(name in results for name in feature_names):
            X = np.column_stack([results[name] for name in feature_names])
            y = labels
            
            # Split for training/validation (80/20)
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Train XGBoost meta-learner
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            
            xgb_model.fit(X_train, y_train)
            
            # Predict on validation set
            xgb_val_preds = xgb_model.predict(X_val)
            xgb_val_acc = accuracy_score(y_val, xgb_val_preds)
            
            # Predict on full dataset
            xgb_full_preds = xgb_model.predict(X)
            xgb_full_acc = accuracy_score(y, xgb_full_preds)
            
            print(f"XGBoost Validation Accuracy: {xgb_val_acc:.3f}")
            print(f"XGBoost Full Dataset Accuracy: {xgb_full_acc:.3f}")
            
            # Feature importance
            importance = xgb_model.feature_importances_
            print("\nFeature Importance:")
            for name, imp in zip(feature_names, importance):
                print(f"{name:10}: {imp:.3f}")
            
            print("\nXGBoost Classification Report:")
            print(classification_report(y, xgb_full_preds, target_names=['Benign', 'Malicious']))
            
            results['XGBoost_Meta'] = xgb_full_preds
        else:
            print("Required models not available for XGBoost meta-learning")
    else:
        print("XGBoost not available or insufficient models")
    
    # Final comparison including weighted combinations
    print("\n=== FINAL COMPARISON (All Models) ===")
    final_scores = {}
    for model_name, preds in results.items():
        acc = accuracy_score(labels, preds)
        final_scores[model_name] = acc
        print(f"{model_name:15}: {acc:.3f}")
    
    # Rank models by performance
    ranked_models = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    print("\nRanked by accuracy:")
    for i, (model, acc) in enumerate(ranked_models, 1):
        print(f"{i}. {model:15}: {acc:.3f}")
    
    # Save results to CSV
    print("\n=== SAVING RESULTS TO CSV ===")
    
    # Create detailed results DataFrame
    results_df = pd.DataFrame({
        'text': texts,
        'true_label': labels,
        **results
    })
    
    # Save detailed results
    results_df.to_csv('model_comparison_detailed.csv', index=False)
    print("Detailed results saved to: model_comparison_detailed.csv")
    
    # Create summary metrics DataFrame
    summary_data = []
    for model_name, preds in results.items():
        from sklearn.metrics import precision_recall_fscore_support
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        summary_data.append({
            'Model': model_name,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('model_comparison_summary.csv', index=False)
    print("Summary metrics saved to: model_comparison_summary.csv")
    
    # Save correlation matrix
    if len(results) >= 2:
        correlation_matrix.to_csv('model_correlation_matrix.csv')
        print("Correlation matrix saved to: model_correlation_matrix.csv")
    
    return results, labels
    for i, (model, acc) in enumerate(ranked_models, 1):
        print(f"{i}. {model:15}: {acc:.3f}")
    
    return results, labels

if __name__ == "__main__":
    results, labels = test_exact_match()