import numpy as np
import pandas as pd
import re
from typing import Dict, Tuple

class NumericFeatureExtractor:
    """
    Extract aggregate numeric scores from text statistics and linguistic features.
    Returns just two numbers per text instead of training a classifier.
    """
    
    def __init__(self):
        pass
    
    def _extract_text_statistics(self, text: str) -> Dict[str, float]:
        """Extract basic text statistics (same as before)."""
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
        """Extract linguistic features (same as before)."""
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
    
    def calculate_text_statistics_score(self, text: str) -> float:
        """
        Calculate aggregate text statistics score (0-1 normalized).
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        stats = self._extract_text_statistics(text)
        
        # Normalize key metrics to 0-1 scale
        length_norm = min(stats['text_length'] / 500, 1.0)  # Normalize by 500 chars
        word_count_norm = min(stats['text_word_count'] / 50, 1.0)  # Normalize by 50 words
        
        # Combine important text statistics with weights
        score = (
            0.2 * length_norm +
            0.15 * word_count_norm +
            0.2 * stats['text_caps_ratio'] +
            0.25 * stats['text_special_char_ratio'] +
            0.1 * stats['text_digit_ratio'] +
            0.1 * min(stats['text_exclamation_marks'] / 5, 1.0)  # Normalize by max 5
        )
        
        return min(score, 1.0)
    
    def calculate_linguistic_features_score(self, text: str) -> float:
        """
        Calculate aggregate linguistic features score (0-1 normalized).
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        ling = self._extract_linguistic_features(text)
        
        # Combine linguistic features with weights based on your classifier results
        # Politeness was most important (19%), so we give it high weight but inverse
        politeness_inverse = max(0, 0.2 - ling['ling_politeness_score']) / 0.2  # Higher score = less polite
        
        score = (
            0.25 * min(ling['ling_command_density'] * 10, 1.0) +  # Scale up command density
            0.20 * politeness_inverse +  # Less politeness = higher score
            0.15 * min(ling['ling_urgency_score'] * 10, 1.0) +   # Scale up urgency
            0.15 * min(ling['ling_temporal_density'] * 10, 1.0) + # Scale up temporal
            0.10 * min(ling['ling_conditional_density'] * 10, 1.0) + # Scale up conditional
            0.05 * ling['ling_has_code_blocks'] +
            0.05 * ling['ling_has_urls'] +
            0.05 * max(0, 0.7 - ling['ling_vocab_diversity']) / 0.7  # Lower diversity = higher score
        )
        
        return min(score, 1.0)
    
    def extract_numeric_features(self, text: str) -> Tuple[float, float]:
        """
        Extract both numeric scores for a single text.
        
        Returns:
            tuple: (text_statistics_score, linguistic_features_score)
        """
        text_stats_score = self.calculate_text_statistics_score(text)
        linguistic_score = self.calculate_linguistic_features_score(text)
        
        return text_stats_score, linguistic_score
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Process dataframe and add numeric feature columns.
        """
        df_copy = df.copy()
        
        print(f"Processing {len(df_copy)} texts for numeric features...")
        
        text_stats_scores = []
        linguistic_scores = []
        
        for i, text in enumerate(df_copy[text_column]):
            if i % 5000 == 0:
                print(f"Processed {i}/{len(df_copy)} texts")
            
            if pd.isna(text) or text == '':
                text_stats_scores.append(0.0)
                linguistic_scores.append(0.0)
            else:
                text_stat, ling_feat = self.extract_numeric_features(str(text))
                text_stats_scores.append(text_stat)
                linguistic_scores.append(ling_feat)
        
        # Add the two numeric columns
        df_copy['text_statistics_numeric'] = text_stats_scores
        df_copy['linguistic_features_numeric'] = linguistic_scores
        
        return df_copy


def get_text_features(text):
    """Simple function to get features for a single text."""
    extractor = NumericFeatureExtractor()
    return extractor.extract_numeric_features(text)
