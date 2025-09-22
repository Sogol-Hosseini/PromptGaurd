import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, KFold, train_test_split
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

def recreate_test_data_with_seed(seed=42, test_size=300):
    """Recreate test data with specific seed for reproducible shuffling"""
    print(f"Recreating test data with seed {seed}...")
    
    # Load the same 3 datasets
    ds1 = load_dataset("xTRam1/safe-guard-prompt-injection")
    ds2 = load_dataset("deepset/prompt-injections")
    ds3 = load_dataset("jayavibhav/prompt-injection")
    
    # Access splits
    test1, test2, test3 = ds1["test"], ds2["test"], ds3["test"]
    
    # Concatenate and shuffle with specified seed
    _test_all_hf = concatenate_datasets([test1, test2, test3]).shuffle(seed=seed)
    
    # Sample specified size
    _test_all_hf = _test_all_hf.select(range(min(test_size, len(_test_all_hf))))
    
    # Convert to pandas with unified columns
    test_df = _df_from_split(_test_all_hf)
    
    return test_df

def evaluate_single_run(texts, labels, run_number=0):
    """Evaluate all models for a single run"""
    results = {}
    
    print(f"  Evaluating {len(texts)} samples...")
    print(f"  Label distribution: {np.bincount(labels)}")
    
    # 1. BERT Model (Fine-tuned)
    print(f"  Run {run_number + 1}: Evaluating BERT...")
    try:
        MODEL_DIR = "models/bert-pi-detector/fine_tuned"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        
        threshold_path = os.path.join(MODEL_DIR, "threshold.txt")
        threshold = float(open(threshold_path).read().strip())
        
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
        
        results['BERT'] = bert_predictions
        print(f"    BERT completed. Accuracy: {accuracy_score(labels, bert_predictions):.3f}")
        
    except Exception as e:
        print(f"    BERT model failed: {e}")
    
    # 2. Regex Model
    print(f"  Run {run_number + 1}: Evaluating Regex...")
    try:
        regex_model = RegexBasedDetector("src/utils/patterns.regex.yaml")
        regex_predictions = []
        for text in texts:
            score = regex_model.score(text)
            pred = 1 if score["level"] in ("warn", "block") else 0
            regex_predictions.append(pred)
        
        results['Regex'] = regex_predictions
        print(f"    Regex completed. Accuracy: {accuracy_score(labels, regex_predictions):.3f}")
        
    except Exception as e:
        print(f"    Regex model failed: {e}")
    
    # 3. Ensemble Model
    print(f"  Run {run_number + 1}: Evaluating Ensemble...")
    try:
        ensemble_model = EnsembleGuard()
        ensemble_predictions = []
        for text in texts:
            decision = ensemble_model.decide(text)
            pred = 1 if decision["decision"] == "block" else 0
            ensemble_predictions.append(pred)
        
        results['Ensemble'] = ensemble_predictions
        print(f"    Ensemble completed. Accuracy: {accuracy_score(labels, ensemble_predictions):.3f}")
        
    except Exception as e:
        print(f"    Ensemble model failed: {e}")
    
    return results

def evaluate_with_statistics(n_runs=5, test_size=300):
    """Run evaluation multiple times and calculate statistics"""
    
    print(f"\n{'='*60}")
    print(f"STATISTICAL EVALUATION WITH {n_runs} RUNS")
    print(f"{'='*60}")
    
    results_across_runs = {
        'BERT': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'Regex': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'Ensemble': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    }
    
    all_predictions = {model: [] for model in results_across_runs.keys()}
    all_labels = []
    
    for run in range(n_runs):
        print(f"\n--- RUN {run + 1}/{n_runs} ---")
        
        # Use different random seed for data shuffling
        test_df = recreate_test_data_with_seed(seed=42 + run, test_size=test_size)
        texts = test_df['text'].tolist()
        labels = test_df['label'].values
        
        # Store labels from first run for later analysis
        if run == 0:
            all_labels = labels
        
        # Evaluate each model
        run_results = evaluate_single_run(texts, labels, run)
        
        # Store metrics for each model
        for model_name, predictions in run_results.items():
            if model_name in results_across_runs:
                # Store predictions for later analysis
                all_predictions[model_name].append(predictions)
                
                # Calculate metrics
                acc = accuracy_score(labels, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, predictions, average='binary', zero_division=0
                )
                
                results_across_runs[model_name]['accuracy'].append(acc)
                results_across_runs[model_name]['precision'].append(precision)
                results_across_runs[model_name]['recall'].append(recall)
                results_across_runs[model_name]['f1'].append(f1)
                
                print(f"  {model_name}: Acc={acc:.3f}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    # Calculate and display statistics
    print(f"\n{'='*60}")
    print("STATISTICAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    statistics = {}
    
    for model_name, metrics in results_across_runs.items():
        if not any(metrics.values()):  # Skip if no results
            continue
            
        model_stats = {}
        print(f"\n{model_name} Results:")
        print("-" * 40)
        
        for metric_name, values in metrics.items():
            if values:  # Check if we have values
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                model_stats[metric_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'values': values
                }
                
                print(f"{metric_name.capitalize():10}: {mean_val:.3f} ± {std_val:.3f} (min: {min_val:.3f}, max: {max_val:.3f})")
                print(f"           Individual runs: {[f'{v:.3f}' for v in values]}")
        
        statistics[model_name] = model_stats
    
    # Model comparison and ranking
    print(f"\n{'='*60}")
    print("MODEL RANKING BY MEAN ACCURACY")
    print(f"{'='*60}")
    
    accuracy_ranking = []
    for model_name, stats in statistics.items():
        if 'accuracy' in stats:
            mean_acc = stats['accuracy']['mean']
            std_acc = stats['accuracy']['std']
            accuracy_ranking.append((model_name, mean_acc, std_acc))
    
    accuracy_ranking.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, mean_acc, std_acc) in enumerate(accuracy_ranking, 1):
        print(f"{i}. {model:15}: {mean_acc:.3f} ± {std_acc:.3f}")
    
    return statistics, all_predictions, all_labels

def bootstrap_evaluation(predictions_dict, true_labels, n_bootstrap=1000):
    """Calculate bootstrap statistics for model predictions"""
    
    print(f"\n{'='*60}")
    print(f"BOOTSTRAP EVALUATION ({n_bootstrap} samples)")
    print(f"{'='*60}")
    
    statistics = {}
    n_samples = len(true_labels)
    
    for model_name, predictions in predictions_dict.items():
        if not predictions:  # Skip if no predictions
            continue
            
        model_stats = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        
        print(f"Running bootstrap for {model_name}...")
        
        for i in range(n_bootstrap):
            if i % 200 == 0:  # Progress indicator
                print(f"  Progress: {i}/{n_bootstrap}")
            
            # Bootstrap sample indices
            boot_indices = np.random.choice(n_samples, n_samples, replace=True)
            
            boot_true = true_labels[boot_indices]
            boot_pred = np.array(predictions)[boot_indices]
            
            # Calculate metrics
            acc = accuracy_score(boot_true, boot_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                boot_true, boot_pred, average='binary', zero_division=0
            )
            
            model_stats['accuracy'].append(acc)
            model_stats['precision'].append(precision)
            model_stats['recall'].append(recall)
            model_stats['f1'].append(f1)
        
        # Calculate statistics
        final_stats = {}
        for metric, values in model_stats.items():
            final_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'median': np.median(values)
            }
        
        statistics[model_name] = final_stats
        
        print(f"\n{model_name} Bootstrap Results:")
        print("-" * 40)
        for metric, stats in final_stats.items():
            print(f"{metric.capitalize():10}: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"           95% CI: [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")
            print(f"           Median: {stats['median']:.3f}")
    
    return statistics

def k_fold_evaluation(k=5, test_size=1000):
    """Perform k-fold cross validation"""
    
    print(f"\n{'='*60}")
    print(f"{k}-FOLD CROSS VALIDATION")
    print(f"{'='*60}")
    
    # Get your full dataset (larger for k-fold)
    test_df = recreate_test_data_with_seed(seed=42, test_size=test_size)
    texts = test_df['text'].tolist()
    labels = test_df['label'].values
    
    print(f"Dataset size: {len(texts)}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = {'BERT': [], 'Regex': [], 'Ensemble': []}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
        print(f"\n--- FOLD {fold + 1}/{k} ---")
        print(f"Validation set size: {len(val_idx)}")
        
        # Get validation data for this fold
        val_texts = [texts[i] for i in val_idx]
        val_labels = labels[val_idx]
        
        # Evaluate models on this fold
        fold_predictions = evaluate_single_run(val_texts, val_labels, fold)
        
        for model_name, predictions in fold_predictions.items():
            if predictions:  # Only if we have predictions
                acc = accuracy_score(val_labels, predictions)
                fold_results[model_name].append(acc)
                print(f"  {model_name} Fold {fold + 1} Accuracy: {acc:.3f}")
    
    # Calculate final statistics
    print(f"\n{'='*60}")
    print("K-FOLD RESULTS SUMMARY")
    print(f"{'='*60}")
    
    kfold_stats = {}
    for model_name, scores in fold_results.items():
        if scores:  # Only if we have scores
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            kfold_stats[model_name] = {
                'mean': mean_score,
                'std': std_score,
                'min': min_score,
                'max': max_score,
                'scores': scores
            }
            
            print(f"{model_name:10}: {mean_score:.3f} ± {std_score:.3f} (min: {min_score:.3f}, max: {max_score:.3f})")
            print(f"           Fold scores: {[f'{s:.3f}' for s in scores]}")
    
    return kfold_stats

def save_detailed_results(statistics, bootstrap_stats, kfold_stats):
    """Save all statistical results to files"""
    
    print(f"\n{'='*60}")
    print("SAVING RESULTS TO FILES")
    print(f"{'='*60}")
    
    # Save multiple runs statistics
    if statistics:
        stats_data = []
        for model_name, metrics in statistics.items():
            for metric_name, values in metrics.items():
                stats_data.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Mean': values['mean'],
                    'Std': values['std'],
                    'Min': values['min'],
                    'Max': values['max'],
                    'Individual_Values': str(values['values'])
                })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv('statistical_evaluation_results.csv', index=False)
        print("Multiple runs statistics saved to: statistical_evaluation_results.csv")
    
    # Save bootstrap results
    if bootstrap_stats:
        bootstrap_data = []
        for model_name, metrics in bootstrap_stats.items():
            for metric_name, values in metrics.items():
                bootstrap_data.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Mean': values['mean'],
                    'Std': values['std'],
                    'Median': values['median'],
                    'CI_Lower': values['ci_lower'],
                    'CI_Upper': values['ci_upper']
                })
        
        bootstrap_df = pd.DataFrame(bootstrap_data)
        bootstrap_df.to_csv('bootstrap_evaluation_results.csv', index=False)
        print("Bootstrap statistics saved to: bootstrap_evaluation_results.csv")
    
    # Save k-fold results
    if kfold_stats:
        kfold_data = []
        for model_name, values in kfold_stats.items():
            kfold_data.append({
                'Model': model_name,
                'Mean': values['mean'],
                'Std': values['std'],
                'Min': values['min'],
                'Max': values['max'],
                'Fold_Scores': str(values['scores'])
            })
        
        kfold_df = pd.DataFrame(kfold_data)
        kfold_df.to_csv('kfold_evaluation_results.csv', index=False)
        print("K-fold statistics saved to: kfold_evaluation_results.csv")

def main():
    """Main function to run all evaluations"""
    
    print("PROMPT INJECTION DETECTION - STATISTICAL EVALUATION")
    print("=" * 80)
    
    # Configuration
    N_RUNS = 5  # Number of runs for statistical evaluation
    TEST_SIZE = 300  # Size of test set for each run
    N_BOOTSTRAP = 1000  # Number of bootstrap samples
    K_FOLDS = 5  # Number of folds for cross-validation
    KFOLD_SIZE = 1000  # Larger dataset for k-fold
    
    # 1. Multiple runs with statistics (RECOMMENDED)
    print("\n" + "="*80)
    print("1. MULTIPLE RUNS EVALUATION")
    print("="*80)
    statistics, all_predictions, all_labels = evaluate_with_statistics(
        n_runs=N_RUNS, 
        test_size=TEST_SIZE
    )
    
    # 2. Bootstrap evaluation on first run
    print("\n" + "="*80)
    print("2. BOOTSTRAP EVALUATION")
    print("="*80)
    if all_predictions and any(all_predictions.values()):
        # Use predictions from first run
        first_run_predictions = {}
        for model_name, pred_list in all_predictions.items():
            if pred_list:  # Check if we have predictions
                first_run_predictions[model_name] = pred_list[0]
        
        bootstrap_stats = bootstrap_evaluation(
            first_run_predictions, 
            all_labels, 
            n_bootstrap=N_BOOTSTRAP
        )
    else:
        bootstrap_stats = {}
        print("No predictions available for bootstrap evaluation")
    
    # 3. K-fold cross validation
    print("\n" + "="*80)
    print("3. K-FOLD CROSS VALIDATION")
    print("="*80)
    kfold_stats = k_fold_evaluation(k=K_FOLDS, test_size=KFOLD_SIZE)
    
    # 4. Save all results
    save_detailed_results(statistics, bootstrap_stats, kfold_stats)
    
    # 5. Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - ALL EVALUATION METHODS")
    print(f"{'='*80}")
    
    print(f"\n1. Multiple Runs ({N_RUNS} runs, {TEST_SIZE} samples each):")
    print("-" * 50)
    if statistics:
        for model_name, metrics in statistics.items():
            if 'accuracy' in metrics:
                acc = metrics['accuracy']
                print(f"{model_name:12}: {acc['mean']:.3f} ± {acc['std']:.3f}")
    
    print(f"\n2. Bootstrap ({N_BOOTSTRAP} samples):")
    print("-" * 50)
    if bootstrap_stats:
        for model_name, metrics in bootstrap_stats.items():
            if 'accuracy' in metrics:
                acc = metrics['accuracy']
                print(f"{model_name:12}: {acc['mean']:.3f} ± {acc['std']:.3f}")
    
    print(f"\n3. K-Fold CV ({K_FOLDS} folds, {KFOLD_SIZE} samples):")
    print("-" * 50)
    if kfold_stats:
        for model_name, values in kfold_stats.items():
            print(f"{model_name:12}: {values['mean']:.3f} ± {values['std']:.3f}")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print("Check the generated CSV files for detailed results.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()