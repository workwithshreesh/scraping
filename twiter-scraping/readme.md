# Trading Signal Analysis System

A comprehensive Python system for collecting, analyzing, and visualizing trading signals from social media data with advanced machine learning and memory-efficient processing.

## Overview

This system transforms raw social media content (tweets) into actionable trading signals using natural language processing, machine learning ensembles, and interactive visualizations. Built specifically for Indian stock market analysis with support for manual CSV input.

## Key Features

- **Text-to-Signal Conversion**: Transform tweet content into 73+ numerical trading features using TF-IDF, sentiment analysis, and custom keyword extraction
- **Memory-Efficient Visualization**: Handle large datasets (1000+ tweets) with smart sampling and interactive Plotly dashboards
- **Signal Aggregation**: Ensemble machine learning models (Random Forest, Ridge, Linear, Voting) with confidence intervals
- **Manual CSV Input**: Full control over data sources with validation and preprocessing
- **Interactive Dashboard**: 6-panel analysis including signal distribution, confidence metrics, and feature importance
- **Indian Market Focus**: Optimized for Nifty, Sensex, and Indian stock symbols

## System Architecture

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  Data Collection │    │   Signal Analysis    │    │   Visualization     │
│                 │    │                      │    │                     │
│ • Twitter Scraper│───▶│ • Text-to-Signal     │───▶│ • Interactive       │
│ • CSV Input     │    │ • ML Aggregation     │    │   Dashboard         │
│ • Validation    │    │ • Confidence Calc    │    │ • Word Clouds       │
└─────────────────┘    └──────────────────────┘    └─────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone or download the files
git clone <your-repo-url>
cd trading-signal-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
pip install plotly wordcloud textblob aiohttp joblib
```

### Usage

1. **Run the complete analysis system:**
```bash
python comprehensive_trading_system.py
```

2. **Enter your CSV filename when prompted:**
```
Enter CSV filename (with .csv extension): your_data.csv
```

3. **View results:**
- Enhanced dataset with signals: `your_data_enhanced_TIMESTAMP.csv`
- Interactive dashboard: `your_data_dashboard.html`
- Model files: `your_data_models_TIMESTAMP.joblib`


## Core Components

### 1. TextToSignalConverter
Converts raw text into numerical trading features:
- **Keyword Analysis**: Bullish/bearish/technical term detection
- **TF-IDF Vectorization**: Document-term matrix with dimensionality reduction
- **Sentiment Scoring**: TextBlob polarity and trading-specific sentiment
- **Feature Engineering**: 73+ features including interaction terms

### 2. MemoryEfficientVisualizer
Creates visualizations for large datasets:
- **Smart Sampling**: Stratified/temporal/random sampling strategies
- **Interactive Dashboards**: 6-panel Plotly visualization
- **Word Cloud Analysis**: Signal-specific text visualization
- **Streaming-Ready**: Handles 10,000+ data points efficiently

### 3. SignalAggregator
Machine learning ensemble for signal prediction:
- **4 Model Ensemble**: Random Forest, Ridge, Linear, Voting Regressor
- **Cross-Validation**: Time-series split validation
- **Confidence Intervals**: Uncertainty quantification
- **Signal Classification**: 7-tier system (STRONG_BUY to STRONG_SELL)

## Input Data Format

### Required CSV Structure
```csv
username,user_id,tweet_id,timestamp,content,likes,retweets,replies,hashtags,url,verified
TradingGuru,123456,789012,2025-09-22T23:00:17,Nifty 50 bullish breakout,25,12,5,#nifty50|#trading,https://...,False
```

### Minimum Requirements
- **content**: Text content (required)
- **timestamp**: ISO format datetime (auto-generated if missing)
- **engagement metrics**: likes, retweets, replies (optional)

## Output Files

### Enhanced Dataset
Original data plus 73+ signal features:
- Keyword counts and densities
- TF-IDF vectors (50 dimensions)
- Sentiment scores
- Model predictions and confidence
- Signal classifications

### Interactive Dashboard
6-panel HTML visualization:
1. Signal distribution over time
2. Signal strength breakdown
3. Confidence vs score correlation
4. Engagement vs sentiment analysis
5. Feature importance ranking
6. Signal volatility timeline

### Model Performance
Cross-validation metrics and feature importance:
```json
{
  "model_performance": {
    "random_forest": 0.1234,
    "ridge_regression": 0.1456,
    "ensemble": 0.1123
  },
  "signal_distribution": {
    "BUY": 234,
    "HOLD": 123,
    "SELL": 89
  }
}
```

## Performance Benchmarks

| Dataset Size | Processing Time | Memory Usage | Recommended RAM |
|--------------|----------------|--------------|-----------------|
| 500 tweets   | 2-3 minutes    | 4 GB         | 8 GB           |
| 1000 tweets  | 3-4 minutes    | 6 GB         | 16 GB          |
| 5000 tweets  | 15-20 minutes  | 12 GB        | 32 GB          |

## Configuration

### Adjust Signal Thresholds
```python
# In SignalAggregator class
self.thresholds = {
    'STRONG_BUY': 0.6,
    'BUY': 0.3,
    'WEAK_BUY': 0.1,
    'HOLD': 0.0,
    'WEAK_SELL': -0.1,
    'SELL': -0.3,
    'STRONG_SELL': -0.6
}
```

### Memory Optimization
```python
# In MemoryEfficientVisualizer
self.sample_size = 5000  # Reduce for lower memory usage

# In TextToSignalConverter
self.max_features = 500  # Reduce TF-IDF features
```

## Development History

### Phase 1: Core Architecture (Initial)
- Built signal aggregation foundation
- Implemented ensemble modeling
- Created basic visualization framework

### Phase 2: Text Processing Enhancement
- Added comprehensive text-to-signal conversion
- Integrated TF-IDF vectorization with dimensionality reduction
- Implemented trading-specific keyword extraction

### Phase 3: Visualization & Memory Optimization
- Developed memory-efficient visualization strategies
- Created interactive Plotly dashboards
- Added smart sampling algorithms

### Phase 4: Integration & User Experience
- Integrated all components into unified system
- Added manual CSV input with validation
- Implemented comprehensive error handling and logging

### Current Version Features
- 73+ signal features per tweet
- 4-model ensemble with confidence intervals
- Interactive 6-panel dashboard
- Memory-efficient processing for large datasets
- Complete pipeline automation with manual data control

## Technical Specifications

### Machine Learning Models
- **Random Forest**: 100 estimators, max_depth=10
- **Ridge Regression**: L2 regularization, alpha=1.0
- **Linear Regression**: OLS baseline
- **Voting Regressor**: Weighted ensemble of above models

### Feature Engineering
- **Text Features**: 50 TF-IDF dimensions + sentiment scores
- **Keyword Features**: 15+ trading-specific categories
- **Interaction Features**: Cross-products of top features
- **Temporal Features**: Hour/day cyclical encoding

### Validation Strategy
- **Time Series Cross-Validation**: 5-fold split preserving temporal order
- **Performance Metrics**: Mean Absolute Error (MAE)
- **Confidence Calculation**: Standard deviation across model predictions

## Troubleshooting

### Common Issues

**Memory Error**
```bash
# Solution: Reduce sample size
python comprehensive_trading_system.py --sample-size 5000
```

**CSV Column Missing**
```bash
# Ensure CSV has 'content' column or rename text column
```

**Poor Signal Quality**
```bash
# Use more diverse trading content with technical and sentiment terms
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Add comprehensive tests for new functionality
4. Ensure backward compatibility
5. Submit pull request with detailed description

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Future Enhancements

- Real-time data streaming integration
- Advanced NLP models (BERT, FinBERT)
- Multi-language support for global markets
- API endpoint for signal serving
- Advanced risk management integration

## Contact

For questions, issues, or contributions, please create an issue in the repository or contact the development team.

---

**Disclaimer**: This system is for educational and research purposes. Trading decisions should always incorporate multiple data sources and professional risk management. Past performance does not guarantee future results.