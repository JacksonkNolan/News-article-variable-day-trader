# Quantitative Market Sentiment Analyzer

### Overview
This project is an automated financial analysis engine that aggregates **technical market data** with **AI-driven sentiment analysis** to generate trading signals. 

Unlike traditional bots that rely solely on price action, this system leverages **Natural Language Processing (NLP)** via the **FinBERT** model to interpret nuance in financial news headlines (e.g., understanding that "inflation" is negative for markets).

### Features
* **Deep Learning NLP:** Utilizes Hugging Face's `ProsusAI/finbert` to classify news sentiment with high accuracy.
* **Macro-Economic Weighting:** Custom algorithm detects and weighs high-impact keywords (e.g., "Fed Rates," "Tariffs," "Recession").
* **Technical Analysis:** Calculates RSI (Relative Strength Index) and SMA (Simple Moving Averages) to validate entry points.
* **Live Data Pipeline:** Fetches real-time market data via `yfinance`.

### Technology Stack
* **Language:** Python 3.10+
* **AI/ML:** PyTorch, Transformers (Hugging Face)
* **Data Processing:** Pandas, NumPy, SciPy
* **Data Source:** Yahoo Finance API

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/market-sentiment-analyzer.git](https://github.com/yourusername/market-sentiment-analyzer.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the analyzer:
    ```bash
    python main.py
    ```

### Disclaimer
**Educational Purpose Only:** This software is designed for academic research and algorithmic development practice. It does not constitute financial advice.

---
*Created by Jackson Nolan*
