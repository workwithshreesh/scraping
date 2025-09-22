import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from textblob import TextBlob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from wordcloud import WordCloud
import logging
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
import json
import joblib
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class TextToSignalConverter:
    
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.tfidf_vectorizer = None
        self.pca_reducer = None
        self.scaler = StandardScaler()
        
        self.trading_keywords = {
            'bullish': ['buy', 'long', 'bullish', 'target', 'breakout', 'rally', 'gain', 'rise', 'up', 'positive', 'strong'],
            'bearish': ['sell', 'short', 'bearish', 'stop', 'breakdown', 'fall', 'drop', 'decline', 'down', 'negative', 'weak'],
            'technical': ['support', 'resistance', 'trend', 'pattern', 'volume', 'rsi', 'macd', 'sma', 'ema'],
            'urgency': ['alert', 'now', 'urgent', 'breaking', 'immediate', 'today', 'quick'],
            'uncertainty': ['maybe', 'could', 'might', 'possibly', 'uncertain', 'risk', 'volatile']
        }
        
    def preprocess_text(self, texts: List[str]) -> List[str]:
        processed_texts = []
        
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            
            text = text.lower()
            
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            text = re.sub(r'@\w+|#', '', text)
            
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            text = ' '.join(text.split())
            
            processed_texts.append(text)
        
        return processed_texts
    
    def extract_keyword_features(self, texts: List[str]) -> pd.DataFrame:
        features = []
        
        for text in texts:
            text_lower = text.lower() if isinstance(text, str) else ""
            
            feature_dict = {}
            
            for category, keywords in self.trading_keywords.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                feature_dict[f'{category}_count'] = count
                feature_dict[f'{category}_density'] = count / max(len(text_lower.split()), 1)
            
            bullish_score = feature_dict['bullish_count'] / max(feature_dict['bullish_count'] + feature_dict['bearish_count'], 1)
            bearish_score = feature_dict['bearish_count'] / max(feature_dict['bullish_count'] + feature_dict['bearish_count'], 1)
            
            feature_dict['keyword_sentiment'] = bullish_score - bearish_score
            feature_dict['keyword_intensity'] = feature_dict['bullish_count'] + feature_dict['bearish_count']
            
            feature_dict['has_technical'] = 1 if feature_dict['technical_count'] > 0 else 0
            
            feature_dict['urgency_ratio'] = feature_dict['urgency_count'] / max(len(text_lower.split()), 1)
            feature_dict['uncertainty_ratio'] = feature_dict['uncertainty_count'] / max(len(text_lower.split()), 1)
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def extract_tfidf_features(self, texts: List[str], n_components: int = 50) -> np.ndarray:
     
        processed_texts = self.preprocess_text(texts)
        
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(processed_texts)
        
        if self.pca_reducer is None:
            self.pca_reducer = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_features = self.pca_reducer.fit_transform(tfidf_matrix.toarray())
        else:
            reduced_features = self.pca_reducer.transform(tfidf_matrix.toarray())
        
        return reduced_features
    
    def extract_sentiment_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract comprehensive sentiment features"""
        features = []
        
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            
            blob = TextBlob(text)
            
            feature_dict = {
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity,
                'text_length': len(text),
                'word_count': len(text.split()),
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
                'number_count': len(re.findall(r'\d+', text))
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def convert_to_signals(self, texts: List[str]) -> pd.DataFrame:
        """Convert texts to comprehensive signal features"""
        print("Converting texts to signal features...")
        
        keyword_features = self.extract_keyword_features(texts)
        sentiment_features = self.extract_sentiment_features(texts)
        tfidf_features = self.extract_tfidf_features(texts)
        
        tfidf_df = pd.DataFrame(
            tfidf_features, 
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        signal_df = pd.concat([keyword_features, sentiment_features, tfidf_df], axis=1)
        
        numerical_cols = signal_df.select_dtypes(include=[np.number]).columns
        signal_df[numerical_cols] = self.scaler.fit_transform(signal_df[numerical_cols])
        
        print(f"Generated {signal_df.shape[1]} signal features from {len(texts)} texts")
        return signal_df

class MemoryEfficientVisualizer:
    """Memory-efficient visualization for large datasets"""
    
    def __init__(self, sample_size: int = 10000):
        self.sample_size = sample_size
        plt.style.use('seaborn-v0_8')
        
    def smart_sample(self, df: pd.DataFrame, strategy: str = 'stratified') -> pd.DataFrame:
        """Intelligently sample data for visualization"""
        if len(df) <= self.sample_size:
            return df
        
        print(f"Sampling {self.sample_size} points from {len(df)} records...")
        
        if strategy == 'random':
            return df.sample(n=self.sample_size, random_state=42)
        
        elif strategy == 'stratified':
            if 'signal_strength' in df.columns:
                return df.groupby('signal_strength').apply(
                    lambda x: x.sample(n=min(len(x), self.sample_size // df['signal_strength'].nunique()), 
                                     random_state=42)
                ).reset_index(drop=True)
        
        elif strategy == 'temporal':
            if 'timestamp' in df.columns:
                df_sorted = df.sort_values('timestamp')
                indices = np.linspace(0, len(df_sorted)-1, self.sample_size, dtype=int)
                return df_sorted.iloc[indices]
        
        return df.sample(n=self.sample_size, random_state=42)
    
    def create_signal_dashboard(self, df: pd.DataFrame, output_file: str = None) -> None:
        print("Creating signal dashboard...")
        
        viz_df = self.smart_sample(df, strategy='stratified')
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Signal Distribution Over Time',
                'Signal Strength Distribution',
                'Confidence vs Score Scatter',
                'Engagement vs Sentiment',
                'Top Features Importance',
                'Signal Volatility Timeline'
            ),
            specs=[[{"secondary_y": True}, {}],
                   [{}, {}],
                   [{}, {}]]
        )
        
        if 'timestamp' in viz_df.columns and 'aggregated_score' in viz_df.columns:
            viz_df['timestamp'] = pd.to_datetime(viz_df['timestamp'])
            time_agg = viz_df.groupby(viz_df['timestamp'].dt.floor('H')).agg({
                'aggregated_score': 'mean',
                'signal_confidence': 'mean'
            }).reset_index()
            
            fig.add_trace(
                go.Scatter(x=time_agg['timestamp'], y=time_agg['aggregated_score'],
                          name='Avg Score', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=time_agg['timestamp'], y=time_agg['signal_confidence'],
                          name='Avg Confidence', line=dict(color='red')),
                row=1, col=1, secondary_y=True
            )
        
        if 'signal_strength' in viz_df.columns:
            signal_counts = viz_df['signal_strength'].value_counts()
            fig.add_trace(
                go.Bar(x=signal_counts.index, y=signal_counts.values,
                      name='Signal Distribution'),
                row=1, col=2
            )
        
        if 'signal_confidence' in viz_df.columns and 'aggregated_score' in viz_df.columns:
            fig.add_trace(
                go.Scatter(x=viz_df['aggregated_score'], y=viz_df['signal_confidence'],
                          mode='markers', name='Confidence vs Score',
                          marker=dict(size=4, opacity=0.6)),
                row=2, col=1
            )
        
        if 'total_engagement' in viz_df.columns and 'net_sentiment' in viz_df.columns:
            fig.add_trace(
                go.Scatter(x=viz_df['net_sentiment'], y=viz_df['total_engagement'],
                          mode='markers', name='Engagement vs Sentiment',
                          marker=dict(size=4, opacity=0.6)),
                row=2, col=2
            )
        
        # 5. Feature Importance (if available)
        if hasattr(self, 'feature_importance') and self.feature_importance:
            top_features = dict(sorted(self.feature_importance.items(), 
                                     key=lambda x: x[1], reverse=True)[:10])
            fig.add_trace(
                go.Bar(x=list(top_features.values()), y=list(top_features.keys()),
                      orientation='h', name='Feature Importance'),
                row=3, col=1
            )
        
        # 6. Signal Volatility
        if 'signal_volatility' in viz_df.columns and 'timestamp' in viz_df.columns:
            vol_agg = viz_df.groupby(viz_df['timestamp'].dt.floor('H'))['signal_volatility'].mean().reset_index()
            fig.add_trace(
                go.Scatter(x=vol_agg['timestamp'], y=vol_agg['signal_volatility'],
                          name='Signal Volatility', line=dict(color='orange')),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Trading Signal Analysis Dashboard",
            showlegend=False
        )
        
        # Save or show
        if output_file:
            fig.write_html(output_file)
            print(f"Dashboard saved to {output_file}")
        else:
            fig.show()
    
    def create_wordcloud_analysis(self, texts: List[str], signal_strengths: List[str] = None) -> None:
        """Create word clouds for different signal categories"""
        print("Creating word cloud analysis...")
        
        # Sample texts if too many
        if len(texts) > self.sample_size:
            indices = np.random.choice(len(texts), self.sample_size, replace=False)
            texts = [texts[i] for i in indices]
            if signal_strengths:
                signal_strengths = [signal_strengths[i] for i in indices]
        
        if signal_strengths:
            # Create separate word clouds for different signals
            unique_signals = list(set(signal_strengths))
            n_signals = len(unique_signals)
            
            fig, axes = plt.subplots(2, (n_signals + 1) // 2, figsize=(15, 10))
            if n_signals == 1:
                axes = [axes]
            elif len(axes.shape) == 1:
                axes = axes.reshape(1, -1)
            
            axes = axes.flatten()
            
            for i, signal in enumerate(unique_signals):
                signal_texts = [texts[j] for j, s in enumerate(signal_strengths) if s == signal]
                combined_text = ' '.join(signal_texts)
                
                if combined_text.strip():
                    wordcloud = WordCloud(width=400, height=300, 
                                        background_color='white').generate(combined_text)
                    axes[i].imshow(wordcloud, interpolation='bilinear')
                    axes[i].set_title(f'{signal} Signals')
                    axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(unique_signals), len(axes)):
                axes[i].axis('off')
        
        else:
            # Single word cloud for all texts
            combined_text = ' '.join(texts)
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white').generate(combined_text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title('All Trading Signals Word Cloud')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

class SignalAggregator:
    """Advanced signal aggregation with confidence intervals"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Signal classification thresholds
        self.thresholds = {
            'STRONG_BUY': 0.6,
            'BUY': 0.3,
            'WEAK_BUY': 0.1,
            'HOLD': 0.0,
            'WEAK_SELL': -0.1,
            'SELL': -0.3,
            'STRONG_SELL': -0.6
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_ensemble_models(self) -> Dict[str, any]:
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'ridge_regression': Ridge(alpha=1.0, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        # Voting ensemble
        ensemble = VotingRegressor(
            estimators=[(name, model) for name, model in models.items()],
            n_jobs=-1
        )
        models['ensemble'] = ensemble
        
        return models
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        print("Training ensemble models...")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        
        models = self.create_ensemble_models()
        model_scores = {}
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            model.fit(X_scaled, y)
            self.models[name] = model
            
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, 
                                      scoring='neg_mean_absolute_error')
            model_scores[name] = -cv_scores.mean()
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                importance = np.ones(len(X.columns))
            
            self.feature_importance[name] = dict(zip(X.columns, importance))
        
        return model_scores
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with confidence intervals"""
        X_scaled = self.scalers['main'].transform(X)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
        
        weights = {'ensemble': 0.4, 'random_forest': 0.3, 'ridge_regression': 0.2, 'linear_regression': 0.1}
        
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            weight = weights.get(name, 0.1)
            ensemble_pred += weight * pred
        
        pred_std = np.std([predictions[name] for name in predictions.keys()], axis=0)
        confidence_intervals = 1.96 * pred_std  # 95% confidence interval
        
        return ensemble_pred, confidence_intervals
    
    def classify_signals(self, scores: np.ndarray, confidence: np.ndarray) -> List[str]:
        """Classify signal strength with confidence adjustment"""
        classifications = []
        
        for score, conf in zip(scores, confidence):
            # Adjust thresholds based on confidence
            conf_factor = min(conf, 1.0)
            adjusted_thresholds = {k: v * (1 - conf_factor * 0.2) for k, v in self.thresholds.items()}
            
            if score >= adjusted_thresholds['STRONG_BUY']:
                classifications.append('STRONG_BUY')
            elif score >= adjusted_thresholds['BUY']:
                classifications.append('BUY')
            elif score >= adjusted_thresholds['WEAK_BUY']:
                classifications.append('WEAK_BUY')
            elif score <= adjusted_thresholds['STRONG_SELL']:
                classifications.append('STRONG_SELL')
            elif score <= adjusted_thresholds['SELL']:
                classifications.append('SELL')
            elif score <= adjusted_thresholds['WEAK_SELL']:
                classifications.append('WEAK_SELL')
            else:
                classifications.append('HOLD')
        
        return classifications

class ComprehensiveTradingSystem:
    """Main system integrating all components"""
    
    def __init__(self):
        self.text_converter = TextToSignalConverter()
        self.visualizer = MemoryEfficientVisualizer()
        self.aggregator = SignalAggregator()
        
    def get_csv_filename(self) -> str:
        while True:
            filename = input("\nEnter CSV filename (with .csv extension): ").strip()
            
            if not filename:
                print("Please enter a filename.")
                continue
            
            if not filename.lower().endswith('.csv'):
                filename += '.csv'
            
            if os.path.exists(filename):
                return filename
            else:
                print(f"File '{filename}' not found.")
                retry = input("Try another filename? (y/n): ").strip().lower()
                if retry != 'y':
                    raise FileNotFoundError(f"CSV file '{filename}' not found.")
    
    def load_and_validate_data(self, filename: str) -> pd.DataFrame:
        print(f"Loading data from {filename}...")
        
        try:
            df = pd.read_csv(filename)
            print(f"Loaded {len(df)} records")
            
            required_cols = ['content']  # Minimum requirement
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                print(f"Available columns: {list(df.columns)}")
                
                text_cols = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['content', 'text', 'message', 'tweet'])]
                
                if text_cols:
                    content_col = text_cols[0]
                    print(f"Using '{content_col}' as content column")
                    df['content'] = df[content_col]
                else:
                    raise ValueError("No suitable text content column found")
            
            # Ensure timestamp column
            if 'timestamp' not in df.columns:
                time_cols = [col for col in df.columns if 'time' in col.lower()]
                if time_cols:
                    df['timestamp'] = pd.to_datetime(df[time_cols[0]])
                else:
                    df['timestamp'] = pd.date_range(start='2025-09-22', periods=len(df), freq='1min')
                    print("Generated synthetic timestamps")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def run_complete_analysis(self):
        print("=" * 50)
        print("COMPREHENSIVE TRADING SIGNAL ANALYSIS SYSTEM")
        print("=" * 50)
        
        try:
            # Step 1: Get CSV filename
            filename = self.get_csv_filename()
            
            # Step 2: Load and validate data
            df = self.load_and_validate_data(filename)
            
            # Step 3: Text-to-Signal Conversion
            print("\n" + "="*30)
            print("STEP 1: TEXT-TO-SIGNAL CONVERSION")
            print("="*30)
            
            signal_features = self.text_converter.convert_to_signals(df['content'].tolist())
            
            # Combine with original data
            analysis_df = pd.concat([df.reset_index(drop=True), signal_features], axis=1)
            
            # Step 4: Signal Aggregation
            print("\n" + "="*30)
            print("STEP 2: SIGNAL AGGREGATION")
            print("="*30)
            
            if 'net_sentiment' in analysis_df.columns:
                target = analysis_df['net_sentiment']
            elif 'textblob_polarity' in analysis_df.columns:
                target = analysis_df['textblob_polarity']
            else:
                target = analysis_df['keyword_sentiment'] if 'keyword_sentiment' in analysis_df.columns else np.random.normal(0, 0.1, len(analysis_df))
            
            # Train models
            feature_cols = signal_features.columns.tolist()
            model_scores = self.aggregator.train_models(signal_features, target)
            
            # Generate predictions
            predictions, confidence_intervals = self.aggregator.predict_with_confidence(signal_features)
            
            analysis_df['aggregated_score'] = predictions
            analysis_df['confidence_interval'] = confidence_intervals
            analysis_df['signal_confidence'] = 1 / (1 + confidence_intervals)
            analysis_df['signal_strength'] = self.aggregator.classify_signals(predictions, confidence_intervals)
            
            # Step 5: Memory-Efficient Visualization
            print("\n" + "="*30)
            print("STEP 3: VISUALIZATION")
            print("="*30)
            
            if self.aggregator.feature_importance:
                avg_importance = {}
                all_features = set()
                for model_imp in self.aggregator.feature_importance.values():
                    all_features.update(model_imp.keys())
                
                for feature in all_features:
                    importance_values = [imp.get(feature, 0) for imp in self.aggregator.feature_importance.values()]
                    avg_importance[feature] = np.mean(importance_values)
                
                self.visualizer.feature_importance = avg_importance
            
            dashboard_file = filename.replace('.csv', '_dashboard.html')
            self.visualizer.create_signal_dashboard(analysis_df, dashboard_file)
            
            if 'content' in analysis_df.columns:
                self.visualizer.create_wordcloud_analysis(
                    analysis_df['content'].tolist(),
                    analysis_df['signal_strength'].tolist()
                )
            
            # Step 6: Export Results
            print("\n" + "="*30)
            print("STEP 4: EXPORT RESULTS")
            print("="*30)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save enhanced dataset
            output_file = filename.replace('.csv', f'_enhanced_{timestamp}.csv')
            analysis_df.to_csv(output_file, index=False, encoding='utf-8')
            
            # Save models
            model_file = filename.replace('.csv', f'_models_{timestamp}.joblib')
            joblib.dump({
                'models': self.aggregator.models,
                'scalers': self.aggregator.scalers,
                'text_converter': self.text_converter,
                'feature_importance': self.aggregator.feature_importance
            }, model_file)
            
            # Create summary report
            summary = {
                'analysis_timestamp': datetime.now().isoformat(),
                'input_file': filename,
                'total_records': len(analysis_df),
                'signal_distribution': analysis_df['signal_strength'].value_counts().to_dict(),
                'average_score': float(analysis_df['aggregated_score'].mean()),
                'average_confidence': float(analysis_df['signal_confidence'].mean()),
                'model_performance': {k: float(v) for k, v in model_scores.items()},
                'top_features': dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]) if 'avg_importance' in locals() else {}
            }
            
            summary_file = filename.replace('.csv', f'_summary_{timestamp}.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Final Summary
            print("\n" + "="*50)
            print("ANALYSIS COMPLETE!")
            print("="*50)
            
            print(f"\nInput: {filename}")
            print(f"Records processed: {len(analysis_df)}")
            print(f"Features generated: {len(feature_cols)}")
            
            print(f"\nSignal Distribution:")
            for signal, count in analysis_df['signal_strength'].value_counts().items():
                percentage = (count / len(analysis_df)) * 100
                print(f"  {signal}: {count} ({percentage:.1f}%)")
            
            print(f"\nModel Performance (Cross-validation MAE):")
            for model, score in model_scores.items():
                print(f"  {model}: {score:.4f}")
            
            print(f"\nOutput Files:")
            print(f"  Enhanced dataset: {output_file}")
            print(f"  Dashboard: {dashboard_file}")
            print(f"  Models: {model_file}")
            print(f"  Summary: {summary_file}")
            
            print(f"\nCurrent Market Signal:")
            latest_signal = analysis_df['signal_strength'].iloc[-1]
            latest_score = analysis_df['aggregated_score'].iloc[-1]
            latest_confidence = analysis_df['signal_confidence'].iloc[-1]
            
            print(f"  Signal: {latest_signal}")
            print(f"  Score: {latest_score:.3f}")
            print(f"  Confidence: {latest_confidence:.3f}")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()

def main():
    system = ComprehensiveTradingSystem()
    system.run_complete_analysis()

if __name__ == "__main__":
    main()