import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Dict, List
import pickle

class SimpleFeatureExtractor:
    """
    Simple feature extractor focusing on text statistics and linguistic patterns.
    Start here before adding more complex features.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = None
        self.feature_names = None
        
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract basic text and linguistic features from a prompt.
        """
        features = {}
        
        # Text Statistics
        text_stats = self._extract_text_statistics(text)
        features.update(text_stats)
        
        # Linguistic Features
        linguistic_features = self._extract_linguistic_features(text)
        features.update(linguistic_features)
        
        # Set feature names on first run
        if self.feature_names is None:
            self.feature_names = list(features.keys())
        
        # Convert to consistent numpy array
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
        return feature_vector
    
    def _extract_text_statistics(self, text: str) -> Dict[str, float]:
        """Extract basic text statistics."""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'text_length': len(text),
            'text_word_count': len(words),
            'text_sentence_count': len(sentences),
            'text_avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'text_avg_sentence_length': len(words) / max(len(sentences), 1),
            'text_caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'text_special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1),
            'text_digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'text_question_marks': text.count('?'),
            'text_exclamation_marks': text.count('!'),
            'text_quotation_marks': text.count('"') + text.count("'"),
            'text_parentheses': text.count('(') + text.count(')'),
            'text_brackets': text.count('[') + text.count(']'),
            'text_newlines': text.count('\n'),
        }
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features."""
        # Command/imperative indicators
        command_words = ['tell', 'show', 'give', 'do', 'stop', 'start', 'ignore', 'forget']
        command_count = sum(1 for word in command_words if word in text.lower())
        
        # Temporal indicators
        temporal_words = ['now', 'then', 'next', 'before', 'after', 'first', 'second', 'finally']
        temporal_count = sum(1 for word in temporal_words if word in text.lower())
        
        # Conditional indicators
        conditional_words = ['if', 'when', 'unless', 'provided', 'assuming']
        conditional_count = sum(1 for word in conditional_words if word in text.lower())
        
        # Politeness indicators
        polite_words = ['please', 'thank', 'could', 'would', 'might', 'sorry']
        polite_count = sum(1 for word in polite_words if word in text.lower())
        
        # Urgency indicators
        urgent_words = ['urgent', 'immediately', 'now', 'asap', 'quick', 'fast']
        urgent_count = sum(1 for word in urgent_words if word in text.lower())
        
        words = text.split()
        word_count = len(words)
        
        return {
            'ling_command_density': command_count / max(word_count, 1),
            'ling_temporal_density': temporal_count / max(word_count, 1),
            'ling_conditional_density': conditional_count / max(word_count, 1),
            'ling_politeness_score': polite_count / max(word_count, 1),
            'ling_urgency_score': urgent_count / max(word_count, 1),
            'ling_vocab_diversity': len(set(words)) / max(word_count, 1),
            'ling_has_code_blocks': 1.0 if '```' in text or '<code>' in text else 0.0,
            'ling_has_urls': 1.0 if re.search(r'https?://', text) else 0.0,
            'ling_has_email': 1.0 if re.search(r'\S+@\S+', text) else 0.0,
            'ling_pronoun_density': len(re.findall(r'\b(i|you|we|they|he|she|it)\b', text.lower())) / max(word_count, 1),
        }
    
    def extract_batch_features(self, texts: List[str]) -> np.ndarray:
        """Extract features for multiple texts."""
        features = []
        for text in texts:
            feature_vector = self.extract_features(text)
            features.append(feature_vector)
        return np.array(features)
    
    def train(self, texts: List[str], labels: List[int]) -> Dict:
        """
        Train the classifier on the extracted features.
        """
        print("Extracting features...")
        X = self.extract_batch_features(texts)
        y = np.array(labels)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Features: {len(self.feature_names)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        print("Training classifier...")
        self.classifier.fit(X_scaled, y)
        
        # Get feature importance
        feature_importance = list(zip(self.feature_names, self.classifier.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Print top features
        print("\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"{i+1:2d}. {feature:25s}: {importance:.4f}")
        
        return {
            'feature_importance': feature_importance,
            'n_features': len(self.feature_names),
            'training_samples': len(texts)
        }
    
    def predict(self, text: str) -> Dict:
        """
        Predict on a single text.
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained yet!")
        
        # Extract and scale features
        features = self.extract_features(text)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        return {
            'text': text[:100] + "..." if len(text) > 100 else text,
            'prediction': int(prediction),
            'injection_probability': probabilities[1],
            'safe_probability': probabilities[0],
            'confidence': max(probabilities)
        }
    
    def evaluate(self, test_texts: List[str], test_labels: List[int]) -> Dict:
        """
        Evaluate on test data.
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained yet!")
        
        print("Evaluating on test data...")
        X_test = self.extract_batch_features(test_texts)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.classifier.predict(X_test_scaled)
        y_proba = self.classifier.predict_proba(X_test_scaled)[:, 1]
        
        # Print results
        print("\nClassification Report:")
        print(classification_report(test_labels, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(test_labels, y_pred))
        
        try:
            auc = roc_auc_score(test_labels, y_proba)
            print(f"\nROC-AUC: {auc:.3f}")
        except:
            auc = None
        
        accuracy = (y_pred == test_labels).mean()
        print(f"Accuracy: {accuracy:.3f}")
        
        return {
            'predictions': y_pred,
            'probabilities': y_proba,
            'accuracy': accuracy,
            'auc': auc
        }
    
    def analyze_features(self, text: str) -> pd.DataFrame:
        """
        Analyze features for a single text.
        """
        features = self.extract_features(text)
        
        df = pd.DataFrame({
            'feature_name': self.feature_names,
            'value': features
        })
        
        # Add categories
        df['category'] = df['feature_name'].apply(lambda x: 'Text Stats' if x.startswith('text_') else 'Linguistic')
        
        return df.sort_values('value', ascending=False)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'scaler': self.scaler,
            'classifier': self.classifier,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.classifier = model_data['classifier']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filepath}")

def test_with_real_data():
    """
    Test with your actual dataset.
    """
    # Load your datasets
    TRAIN_PATH = "/Users/kamyartaeb/Desktop/VScode/PromptGaurd/src/Scripts/train_dfs/total_train.csv"
    TEST_PATH = "/Users/kamyartaeb/Desktop/VScode/PromptGaurd/src/test_dfs/total_test.csv"
    
    print("Loading your datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print(f"Train dataset: {len(train_df)} samples")
    print(f"Test dataset: {len(test_df)} samples")
    print(f"Train columns: {list(train_df.columns)}")
    print(f"Test columns: {list(test_df.columns)}")
    
    # Clean data
    train_df = train_df.dropna(subset=['text', 'label'])
    test_df = test_df.dropna(subset=['text', 'label'])
    
    # Sample reasonable amounts for testing
    TRAIN_SAMPLE_SIZE = 100000  # Use 10K for training
    TEST_SAMPLE_SIZE = 20000    # Use 2K for testing
    
    if len(train_df) > TRAIN_SAMPLE_SIZE:
        train_sample = train_df.sample(n=TRAIN_SAMPLE_SIZE, random_state=42)
    else:
        train_sample = train_df
    
    if len(test_df) > TEST_SAMPLE_SIZE:
        test_sample = test_df.sample(n=TEST_SAMPLE_SIZE, random_state=42)
    else:
        test_sample = test_df
    
    print(f"\nUsing {len(train_sample)} training samples")
    print(f"Using {len(test_sample)} test samples")
    
    # Check label distribution
    print(f"\nTrain label distribution:")
    print(train_sample['label'].value_counts())
    print(f"\nTest label distribution:")
    print(test_sample['label'].value_counts())
    
    # Initialize and train
    extractor = SimpleFeatureExtractor()
    
    print("\n" + "="*60)
    print("TRAINING FEATURE CLASSIFIER")
    print("="*60)
    
    # Train
    train_texts = train_sample['text'].tolist()
    train_labels = train_sample['label'].tolist()
    
    results = extractor.train(train_texts, train_labels)
    
    print("\n" + "="*60)
    print("EVALUATING ON TEST DATA")  
    print("="*60)
    
    # Evaluate
    test_texts = test_sample['text'].tolist()
    test_labels = test_sample['label'].tolist()
    
    test_results = extractor.evaluate(test_texts, test_labels)
    
    # Show some example predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    # Get a few examples of each class
    safe_examples = test_sample[test_sample['label'] == 0]['text'].head(3).tolist()
    injection_examples = test_sample[test_sample['label'] == 1]['text'].head(3).tolist()
    
    print("\nSafe examples:")
    for text in safe_examples:
        pred = extractor.predict(text)
        predicted = "Injection" if pred['prediction'] == 1 else "Safe"
        print(f"Text: {pred['text']}")
        print(f"Predicted: {predicted}, Confidence: {pred['confidence']:.3f}")
        print("-" * 40)
    
    print("\nInjection examples:")
    for text in injection_examples:
        pred = extractor.predict(text)
        predicted = "Injection" if pred['prediction'] == 1 else "Safe"
        print(f"Text: {pred['text']}")
        print(f"Predicted: {predicted}, Confidence: {pred['confidence']:.3f}")
        print("-" * 40)
    
    # Save the model
    extractor.save_model("linguistic.pkl")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Training samples: {len(train_sample)}")
    print(f"Test samples: {len(test_sample)}")
    print(f"Features used: {len(extractor.feature_names)}")
    print(f"Test accuracy: {test_results['accuracy']:.3f}")
    if test_results['auc']:
        print(f"Test ROC-AUC: {test_results['auc']:.3f}")
    
    return extractor, test_results

if __name__ == "__main__":
    test_with_real_data()