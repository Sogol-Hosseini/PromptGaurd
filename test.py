import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from datasets import load_dataset, concatenate_datasets
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import login
from src.modules.rules_regex import RegexBasedDetector
from src.modules.ensemble_guard import EnsembleGuard, MODEL_DIR

# Authenticate with Hugging Face
login(token="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")

# Configuration
BERT_MODEL_PATH = "models/bert-pi-detector/fine_tuned"
# CSV_PATH = "hf://datasets/qualifire/prompt-injections-benchmark/test.csv"

class ModelComparison:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_test_datasets(self):
        """Load and combine all test datasets"""
        print("Loading test datasets...")
        
        # Load individual datasets
        ds1 = load_dataset("deepset/prompt-injections")
        ds2 = load_dataset("jayavibhav/prompt-injection")
        
        # Access test splits
        test1 = ds1["test"]
        test2 = ds2["test"]
        
        return {
            'deepset': test1,
            'jayavibhav': test2
        }
    
    def setup_bert_model(self):
        """Setup your fine-tuned BERT model"""
        print("Loading fine-tuned BERT model...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
            
            # Set max length for consistency
            if hasattr(tokenizer, 'model_max_length'):
                tokenizer.model_max_length = 256
            
            pipeline = TextClassificationPipeline(
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                top_k=None,
            )
            
            return pipeline
            
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            return None
    
    def setup_regex_model(self):
        """Setup Regex-based model"""
        print("Loading Regex model...")
        try:
            return RegexBasedDetector("src/utils/patterns.regex.yaml")
        except Exception as e:
            print(f"Error loading Regex model: {e}")
            return None
    
    def setup_lm_model(self):
        """Setup Language Model"""
        print("Loading LM model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            tokenizer.model_max_length = 256
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            
            # Load threshold
            threshold = 0.1
            threshold_path = os.path.join(MODEL_DIR, "threshold.txt")
            if os.path.exists(threshold_path):
                threshold = float(open(threshold_path).read().strip())
            
            pipeline = TextClassificationPipeline(
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                top_k=None,
            )
            
            return pipeline, threshold
            
        except Exception as e:
            print(f"Error loading LM model: {e}")
            return None, None
    
    def setup_ensemble_model(self):
        """Setup Ensemble model"""
        print("Loading Ensemble model...")
        try:
            return EnsembleGuard(model_dir=MODEL_DIR)
        except Exception as e:
            print(f"Error loading Ensemble model: {e}")
            return None
    
    def predict_bert(self, texts, pipeline):
        """Make predictions with BERT model"""
        predictions = []
        probabilities = []
        
        for text in texts:
            try:
                result = pipeline(text, truncation=True, max_length=256)
                
                # Extract probability for positive class (injection)
                if isinstance(result, list) and len(result) > 0:
                    scores = result[0] if isinstance(result[0], list) else result
                    # Find the score for label "1" or "LABEL_1" (injection)
                    prob = next(
                        (s["score"] for s in scores if s["label"].endswith("1")), 
                        0.5
                    )
                    probabilities.append(prob)
                    predictions.append(1 if prob >= 0.5 else 0)
                else:
                    probabilities.append(0.5)
                    predictions.append(0)
                    
            except Exception as e:
                print(f"Error predicting with BERT: {e}")
                probabilities.append(0.5)
                predictions.append(0)
        
        return np.array(predictions), np.array(probabilities)
    
    def predict_regex(self, texts, regex_model):
        """Make predictions with Regex model"""
        predictions = []
        for text in texts:
            try:
                score = regex_model.score(text)
                pred = 1 if score["level"] in ("warn", "block") else 0
                predictions.append(pred)
            except Exception as e:
                print(f"Error predicting with Regex: {e}")
                predictions.append(0)
        
        return np.array(predictions)
    
    def predict_lm(self, texts, lm_pipeline, threshold):
        """Make predictions with LM model"""
        predictions = []
        probabilities = []
        
        for text in texts:
            try:
                scores = lm_pipeline(text, truncation=True, max_length=256)[0]
                prob = next(s["score"] for s in scores if s["label"].endswith("1"))
                probabilities.append(prob)
                predictions.append(1 if prob >= threshold else 0)
            except Exception as e:
                print(f"Error predicting with LM: {e}")
                probabilities.append(0.5)
                predictions.append(0)
        
        return np.array(predictions), np.array(probabilities)
    
    def predict_ensemble(self, texts, ensemble_model):
        """Make predictions with Ensemble model"""
        predictions = []
        for text in texts:
            try:
                decision = ensemble_model.decide(text)
                pred = 1 if decision["decision"] == "block" else 0
                predictions.append(pred)
            except Exception as e:
                print(f"Error predicting with Ensemble: {e}")
                predictions.append(0)
        
        return np.array(predictions)
    
    def evaluate_on_dataset(self, texts, labels, dataset_name):
        """Evaluate all models on a given dataset"""
        print(f"\nEvaluating on {dataset_name}...")
        
        results = {}
        
        # BERT Model
        if 'bert' in self.models and self.models['bert'] is not None:
            y_bert, probs_bert = self.predict_bert(texts, self.models['bert'])
            results['BERT (Fine-tuned)'] = {
                'predictions': y_bert,
                'probabilities': probs_bert
            }
        
        # Regex Model
        if 'regex' in self.models and self.models['regex'] is not None:
            y_regex = self.predict_regex(texts, self.models['regex'])
            results['Regex Only'] = {
                'predictions': y_regex,
                'probabilities': None
            }
        
        # LM Model
        if 'lm' in self.models and self.models['lm'][0] is not None:
            y_lm, probs_lm = self.predict_lm(texts, self.models['lm'][0], self.models['lm'][1])
            results['LM Only'] = {
                'predictions': y_lm,
                'probabilities': probs_lm
            }
        
        # Ensemble Model
        if 'ensemble' in self.models and self.models['ensemble'] is not None:
            y_ensemble = self.predict_ensemble(texts, self.models['ensemble'])
            results['Ensemble (Regex + LM)'] = {
                'predictions': y_ensemble,
                'probabilities': None
            }
        
        return results
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        return {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def print_detailed_results(self, y_true, results, dataset_name):
        """Print detailed results for each model"""
        print(f"\n{'='*50}")
        print(f"RESULTS FOR {dataset_name.upper()}")
        print(f"{'='*50}")
        
        metrics_summary = []
        
        for model_name, preds in results.items():
            y_pred = preds['predictions']
            
            print(f"\n=== {model_name} ===")
            print(classification_report(y_true, y_pred, digits=3))
            print("Confusion Matrix:")
            print(confusion_matrix(y_true, y_pred))
            
            # Calculate and store metrics
            metrics = self.calculate_metrics(y_true, y_pred, model_name)
            metrics_summary.append(metrics)
        
        return metrics_summary
    
    def create_comparison_table(self, all_metrics, dataset_names):
        """Create a comparison table across all datasets"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE COMPARISON ACROSS ALL DATASETS")
        print(f"{'='*80}")
        
        # Create DataFrame for better visualization
        df_results = pd.DataFrame()
        
        for dataset_idx, dataset_name in enumerate(dataset_names):
            if dataset_idx < len(all_metrics):
                df_temp = pd.DataFrame(all_metrics[dataset_idx])
                df_temp['dataset'] = dataset_name
                df_results = pd.concat([df_results, df_temp], ignore_index=True)
        
        # Pivot table for better comparison
        pivot_metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for metric in pivot_metrics:
            print(f"\n{metric.upper()} Comparison:")
            pivot = df_results.pivot(index='model', columns='dataset', values=metric)
            print(pivot.round(3))
            
            # Calculate average across datasets
            if len(dataset_names) > 1:
                pivot['Average'] = pivot.mean(axis=1)
                print(f"\nAverage {metric}:")
                print(pivot['Average'].round(3).sort_values(ascending=False))
        
        return df_results
    
    def plot_comparison(self, df_results):
        """Create visualization plots"""
        try:
            plt.figure(figsize=(15, 10))
            
            # F1 Score comparison
            plt.subplot(2, 2, 1)
            sns.barplot(data=df_results, x='model', y='f1', hue='dataset')
            plt.title('F1 Score Comparison')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Accuracy comparison
            plt.subplot(2, 2, 2)
            sns.barplot(data=df_results, x='model', y='accuracy', hue='dataset')
            plt.title('Accuracy Comparison')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Precision comparison
            plt.subplot(2, 2, 3)
            sns.barplot(data=df_results, x='model', y='precision', hue='dataset')
            plt.title('Precision Comparison')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Recall comparison
            plt.subplot(2, 2, 4)
            sns.barplot(data=df_results, x='model', y='recall', hue='dataset')
            plt.title('Recall Comparison')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def run_complete_evaluation(self):
        """Run the complete evaluation pipeline"""
        print("Starting comprehensive model evaluation...")
        
        # Load all models
        self.models['bert'] = self.setup_bert_model()
        self.models['regex'] = self.setup_regex_model()
        self.models['lm'] = self.setup_lm_model()
        self.models['ensemble'] = self.setup_ensemble_model()
        
        # Load test datasets
        test_datasets = self.load_test_datasets()
        
        all_metrics = []
        dataset_names = []
        
        # Evaluate on each dataset
        for dataset_name, dataset in test_datasets.items():
            dataset_names.append(dataset_name)
            
            # Handle HuggingFace datasets - adjust column names as needed
            if 'text' in dataset.column_names:
                texts = dataset['text']
            elif 'prompt' in dataset.column_names:
                texts = dataset['prompt']
            else:
                print(f"Warning: Could not find text column in {dataset_name}")
                continue
            
            if 'label' in dataset.column_names:
                labels = np.array(dataset['label'])
            elif 'is_injection' in dataset.column_names:
                labels = np.array(dataset['is_injection'])
            else:
                print(f"Warning: Could not find label column in {dataset_name}")
                continue
            
            # Evaluate models on this dataset
            results = self.evaluate_on_dataset(texts, labels, dataset_name)
            
            # Print detailed results
            metrics = self.print_detailed_results(labels, results, dataset_name)
            all_metrics.append(metrics)
        
        # Create comprehensive comparison
        df_results = self.create_comparison_table(all_metrics, dataset_names)
        
        # Create visualizations
        self.plot_comparison(df_results)
        
        print("\nEvaluation completed!")
        return df_results

def main():
    """Main execution function"""
    evaluator = ModelComparison()
    results_df = evaluator.run_complete_evaluation()
    
    # Save results to CSV
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\nResults saved to 'model_comparison_results.csv'")

if __name__ == "__main__":
    main()