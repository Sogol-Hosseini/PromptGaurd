import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
from tabulate import tabulate

class MultiDatasetClassifierComparison:
    """
    Compare multiple classifiers across multiple datasets and display in table format.
    """
    
    def __init__(self):
        self.results = {}
        self.classifiers_config = {
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss'
                ),
                'needs_scaling': False
            },
            'Random Forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    random_state=42
                ),
                'needs_scaling': False
            },
            'Ridge': {
                'model': RidgeClassifier(alpha=1.0, random_state=42),
                'needs_scaling': True
            }
        }
    
    def load_dataset(self, file_path, dataset_name):
        """
        Load and prepare a single dataset.
        """
        print(f"Loading {dataset_name} from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Check required columns
            required_cols = ['text', 'label']
            feature_cols = ['maliciousness_score', 'prompt_inj_score', 'text_statistics', 'linguistic_features']
            
            if not all(col in df.columns for col in required_cols):
                print(f"Missing required columns in {dataset_name}")
                return None
            
            # Use available feature columns
            available_features = [col for col in feature_cols if col in df.columns]
            
            if not available_features:
                print(f"No feature columns found in {dataset_name}")
                return None
            
            print(f"  Shape: {df.shape}")
            print(f"  Features: {available_features}")
            print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
            
            # Prepare features and labels
            X = df[available_features].fillna(0)
            y = df['label']
            
            return {
                'name': dataset_name,
                'X': X,
                'y': y,
                'features': available_features,
                'size': len(df)
            }
            
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None
    
    def evaluate_classifier_on_dataset(self, classifier_name, classifier_config, dataset):
        """
        Train and evaluate a single classifier on a single dataset.
        """
        X, y = dataset['X'], dataset['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Prepare classifier
        model = classifier_config['model']
        scaler = StandardScaler() if classifier_config['needs_scaling'] else None
        
        # Scale if needed
        if scaler:
            X_train_processed = scaler.fit_transform(X_train)
            X_test_processed = scaler.transform(X_test)
        else:
            X_train_processed = X_train
            X_test_processed = X_test
        
        # Train
        model.fit(X_train_processed, y_train)
        
        # Predict
        y_pred = model.predict(X_test_processed)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_processed)[:, 1]
        else:
            # For Ridge classifier
            decision_scores = model.decision_function(X_test_processed)
            y_proba = 1 / (1 + np.exp(-decision_scores))  # Sigmoid
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'test_size': len(y_test)
        }
        
        return metrics
    
    def run_comparison(self, dataset_paths):
        """
        Run comparison across all classifiers and datasets.
        """
        print("="*80)
        print("MULTI-DATASET CLASSIFIER COMPARISON")
        print("="*80)
        
        # Load all datasets
        datasets = []
        for path, name in dataset_paths:
            dataset = self.load_dataset(path, name)
            if dataset:
                datasets.append(dataset)
        
        if not datasets:
            print("No valid datasets loaded!")
            return
        
        print(f"\nSuccessfully loaded {len(datasets)} datasets")
        
        # Run experiments
        for dataset in datasets:
            print(f"\n{'='*60}")
            print(f"EVALUATING ON DATASET: {dataset['name']}")
            print(f"{'='*60}")
            
            dataset_results = {}
            
            for classifier_name, classifier_config in self.classifiers_config.items():
                print(f"Training {classifier_name}...")
                
                try:
                    metrics = self.evaluate_classifier_on_dataset(
                        classifier_name, classifier_config, dataset
                    )
                    dataset_results[classifier_name] = metrics
                    
                    print(f"  Accuracy: {metrics['accuracy']:.3f}")
                    print(f"  F1: {metrics['f1']:.3f}")
                    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    dataset_results[classifier_name] = None
            
            self.results[dataset['name']] = dataset_results
    
    def create_results_table(self, metric='accuracy'):
        """
        Create a formatted table of results for a specific metric.
        """
        if not self.results:
            print("No results available. Run comparison first.")
            return
        
        # Prepare data for table
        datasets = list(self.results.keys())
        classifiers = list(self.classifiers_config.keys())
        
        # Create table data
        table_data = []
        
        for classifier in classifiers:
            row = [classifier]
            for dataset in datasets:
                if (dataset in self.results and 
                    classifier in self.results[dataset] and 
                    self.results[dataset][classifier] is not None):
                    value = self.results[dataset][classifier][metric]
                    row.append(f"{value:.3f}")
                else:
                    row.append("N/A")
            table_data.append(row)
        
        # Create headers
        headers = ['Classifier'] + datasets
        
        # Print table
        print(f"\n{metric.upper()} RESULTS")
        print("="*60)
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        return table_data, headers
    
    def create_comprehensive_table(self):
        """
        Create a comprehensive table with all metrics.
        """
        if not self.results:
            print("No results available.")
            return
        
        print("\nCOMPREHENSIVE RESULTS TABLE")
        print("="*100)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for metric in metrics:
            self.create_results_table(metric)
            print()
    
    def create_summary_dataframe(self):
        """
        Create a pandas DataFrame with all results for easy analysis.
        """
        summary_data = []
        
        for dataset_name, dataset_results in self.results.items():
            for classifier_name, metrics in dataset_results.items():
                if metrics is not None:
                    row = {
                        'Dataset': dataset_name,
                        'Classifier': classifier_name,
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1': metrics['f1'],
                        'ROC_AUC': metrics['roc_auc'],
                        'Test_Size': metrics['test_size']
                    }
                    summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        return df
    
    def print_best_performers(self):
        """
        Print best performing classifier for each dataset and metric.
        """
        print("\nBEST PERFORMERS BY DATASET")
        print("="*60)
        
        metrics = ['accuracy', 'f1', 'roc_auc']
        
        for dataset_name, dataset_results in self.results.items():
            print(f"\nDataset: {dataset_name}")
            print("-" * 30)
            
            for metric in metrics:
                best_score = 0
                best_classifier = "None"
                
                for classifier_name, results in dataset_results.items():
                    if results and results[metric] > best_score:
                        best_score = results[metric]
                        best_classifier = classifier_name
                
                print(f"Best {metric:8s}: {best_classifier:12s} ({best_score:.3f})")

def main():
    """
    Main function to run the comparison.
    """
    # Define your three datasets
    dataset_paths = [
        ("src/test_dfs/processed_dataset1.csv", "Dataset 1"),
        ("src/test_dfs/processed_dataset2.csv", "Dataset 2"), 
        ("src/test_dfs/processed_dataset3.csv", "Dataset 3")
    ]
    
    # Update these paths to match your actual files
    # You can also use the same file with different subsets
    actual_datasets = [
        ("src/test_dfs/processed_total_test.csv", "Total Test"),
        # Add your other datasets here
        # ("path/to/dataset2.csv", "Dataset 2"),
        # ("path/to/dataset3.csv", "Dataset 3"),
    ]
    
    # Initialize comparison
    comparison = MultiDatasetClassifierComparison()
    
    # Run comparison
    comparison.run_comparison(actual_datasets)
    
    # Display results
    comparison.create_comprehensive_table()
    comparison.print_best_performers()
    
    # Create summary DataFrame
    df_summary = comparison.create_summary_dataframe()
    print("\nSUMMARY DATAFRAME:")
    print(df_summary.to_string(index=False))
    
    # Save results
    df_summary.to_csv("classifier_comparison_results.csv", index=False)
    print("\nResults saved to: classifier_comparison_results.csv")

# For Jupyter notebook usage
def compare_classifiers_on_datasets(dataset_paths):
    """
    Convenience function for Jupyter notebook.
    
    Args:
        dataset_paths: List of (file_path, dataset_name) tuples
    
    Returns:
        DataFrame with all results
    """
    comparison = MultiDatasetClassifierComparison()
    comparison.run_comparison(dataset_paths)
    comparison.create_comprehensive_table()
    comparison.print_best_performers()
    
    return comparison.create_summary_dataframe()

if __name__ == "__main__":
    main()