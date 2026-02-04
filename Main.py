"""
Financial Sentiment & Market Analysis Engine
--------------------------------------------
Author: Jackson Nolan
Description: 
    A quantitative analysis tool that leverages BERT-based NLP (FinBERT) 
    to assess market sentiment from news headlines and correlates it 
    with technical indicators (RSI, SMA) to generate trading signals.

Disclaimer: 
    For educational and research purposes only. Not financial advice.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from transformers import pipeline

# Configuration
# ------------------------------------------------------------------
MACRO_WEIGHTS = {
    'inflation': -0.15, 'recession': -0.25, 'rate hike': -0.20,
    'fed': -0.05, 'cut': 0.10, 'growth': 0.15, 'earnings': 0.05,
    'tariff': -0.10, 'election': -0.05, 'stimulus': 0.20,
    'unemployment': -0.10, 'jobs': 0.10, 'crisis': -0.30
}

# Technical Thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Suppress warnings for cleaner console output
warnings.filterwarnings("ignore")

class MarketAnalyzer:
    """
    Main class for fetching financial data and running sentiment analysis.
    """
    
    def __init__(self):
        print(">> Initializing Neural Networks (FinBERT)...")
        # Load FinBERT: Specialized NLP model for financial text
        try:
            self.sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert", device=-1)
        except Exception as e:
            print(f"Error loading AI model: {e}")
            self.sentiment_pipeline = None

    def fetch_technicals(self, ticker: str) -> pd.DataFrame:
        """
        Retrieves 6 months of historical price data and calculates 
        Simple Moving Averages (SMA) and Relative Strength Index (RSI).
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="6mo")
            
            if df.empty:
                return None

            # Calculate Moving Averages
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()

            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # Avoid division by zero issues
            rs = gain / loss.replace(0, np.nan)
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI'] = df['RSI'].fillna(50) # Neutral fill for initial NaNs

            return df
        except Exception as e:
            print(f"Error fetching technical data for {ticker}: {e}")
            return None

    def analyze_sentiment(self, ticker: str):
        """
        Fetches recent news and calculates a composite sentiment score
        based on AI inference (FinBERT) and keyword heuristic analysis.
        
        Returns:
            tuple: (sentiment_score, macro_score)
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return 0.0, 0.0
            
            sentiment_total = 0
            macro_total = 0
            count = 0
            
            print(f"   > Scanning {len(news)} articles for {ticker}...")

            for article in news:
                title = article.get('title', '')
                if not title:
                    continue
                
                # --- UNCOMMENT THE LINE BELOW TO DEBUG HEADLINES ---
                # print(f"     HEADLINE: {title[:60]}...") 

                # 1. AI Inference
                if self.sentiment_pipeline:
                    try:
                        result = self.sentiment_pipeline(title)[0]
                        score = result['score']
                        label = result['label'].lower() # FIX: Ensure label is lowercase
                        
                        if label == 'positive':
                            sentiment_total += score
                        elif label == 'negative':
                            sentiment_total -= score
                        # Neutral contributes 0.0
                    except:
                        pass # Skip if inference fails

                # 2. Macro Keyword Scan
                title_lower = title.lower()
                for word, weight in MACRO_WEIGHTS.items():
                    if word in title_lower:
                        macro_total += weight
                        print(f"     [MACRO EVENT] Found '{word}' in: {title[:40]}...")

                count += 1

            avg_sentiment = sentiment_total / count if count > 0 else 0
            return avg_sentiment, macro_total

        except Exception as e:
            print(f"Error analyzing news for {ticker}: {e}")
            return 0.0, 0.0

    def generate_signal(self, ticker: str):
        """
        Aggregates Technical and Fundamental data to output a decision signal.
        """
        print(f"\n--- Analyzing Asset: {ticker} ---")
        
        df = self.fetch_technicals(ticker)
        if df is None:
            print("   [!] Insufficient data.")
            return

        # Extract latest metrics
        price = df['Close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        sma50 = df['SMA_50'].iloc[-1]

        # Technical Scoring
        tech_score = 0
        if rsi < RSI_OVERSOLD:
            tech_score += 1.0 # Oversold signal
        elif rsi > RSI_OVERBOUGHT:
            tech_score -= 1.0 # Overbought signal
        
        if price > sma50:
            tech_score += 0.5 # Bullish trend
        else:
            tech_score -= 0.5 # Bearish trend

        # Sentiment Scoring
        ai_score, macro_score = self.analyze_sentiment(ticker)

        # Weighted Composite Score
        # 40% Technicals, 40% AI Sentiment, 20% Macro
        final_score = (tech_score * 0.4) + (ai_score * 2.0) + (macro_score * 2.0)

        # Signal Determination
        if final_score > 0.5:
            decision = "STRONG BUY"
            color = "\033[92m" # Green
        elif final_score > 0.2:
            decision = "BUY"
            color = "\033[92m"
        elif final_score < -0.5:
            decision = "STRONG SELL"
            color = "\033[91m" # Red
        elif final_score < -0.2:
            decision = "SELL"
            color = "\033[91m"
        else:
            decision = "HOLD / NEUTRAL"
            color = "\033[93m" # Yellow/White

        print(f"   Price: ${price:.2f}")
        print(f"   Indicators: RSI={rsi:.1f} | SMA50=${sma50:.2f}")
        print(f"   Sentiment: AI={ai_score:.3f} | Macro={macro_score:.3f}")
        print(f"   {color}SIGNAL: {decision} (Confidence: {final_score:.3f})\033[0m")

if __name__ == "__main__":
    # Watchlist for analysis
    WATCHLIST = ["AAPL", "NVDA", "JPM", "SPY"]
    
    bot = MarketAnalyzer()
    
    for symbol in WATCHLIST:
        bot.generate_signal(symbol)
        
    print("\nAnalysis Complete.")
