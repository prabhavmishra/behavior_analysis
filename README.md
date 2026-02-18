# Trader Behavior vs. Market Sentiment Analysis (Hyperliquid)

## Project Overview

This project analyzes the correlation between Bitcoin Market Sentiment (Fear/Greed Index) and individual trader performance on the Hyperliquid exchange. The goal is to uncover behavioral patterns and provide actionable trading strategies based on data-driven insights.

## Methodology

1. **Data Alignment:** Merged daily sentiment classifications with historical trade execution data by normalizing timestamps.
2. **Feature Engineering:** Calculated Daily PnL, Win Rate, Long/Short Ratio, and Average Trade Size.
3. **Segmentation:** Categorized traders into **Frequent** (top 50% by volume) and **Infrequent** (bottom 50% by volume) using median activity thresholds.
4. **Analysis:** Evaluated how performance metrics shift across the five sentiment categories (Extreme Fear to Extreme Greed).

---

## Key Insights

### 1. The Fear Efficiency Gap

The analysis shows a clear performance decay during periods of high market stress.

* **Insight:** Average **Win Rates** peak at **~38.6%** during Extreme Greed but drop to **~32.9%** during Extreme Fear.
* **Visualization:** Refer to `win_rate_sentiment.png`.

### 2. Panic Over-trading Bias

Traders exhibit a "fight or flight" response during market downturns.

* **Insight:** Frequent traders increase their daily trade count by nearly **60%** during Extreme Fear compared to Greed, despite the lower win rate. This indicates "revenge trading" or high-frequency attempts to catch a market bottom.
* **Visualization:** Refer to `behavior_boxplot.png`.

### 3. Segmented Capital Resilience

* **Insight:** Infrequent "Swing" traders maintain more stable PnL profiles by scaling down their trade sizes during Extreme Greed, whereas Frequent traders tend to over-leverage, leading to volatile PnL swings during sentiment shifts.
* **Visualization:** Refer to `segment_pnl_sentiment.png`.

---

## Actionable Recommendations

### Strategy 1: The "Fear Circuit Breaker"

* **Rule:** During **Extreme Fear** (Sentiment < 25), accounts in the Frequent segment should implement an automated **30% reduction in daily trade limits**.
* **Reasoning:** Historical data shows that increased activity in this sentiment results in a negative PnL correlation due to decreased win rates.

### Strategy 2: The "Greed Momentum" Multiplier

* **Rule:** During **Greed or Extreme Greed**, traders should prioritize trend-following strategies with a **15% increase in position sizing**, as the environment supports higher win rates for directional trades.
* **Reasoning:** Exploiting the higher win rate in euphoric markets can maximize capital efficiency.

---

## Technical Deliverables

### Generated CSVs

* `processed_trades_sentiment.csv`: The master trade log with daily sentiment labels.
* `final_trade_analysis.csv`: Daily aggregated metrics per trader.
* `trader_segments.csv`: Categorization of unique accounts.
* `feature_importance.csv`: Ranking of factors (size, bias, frequency) driving profitability.

### Key Visualizations

1. `win_rate_sentiment.png`: Win Rate trends vs. Market Mood.
2. `behavior_boxplot.png`: Trade frequency distribution per segment.
3. `segment_pnl_sentiment.png`: Average Daily PnL comparison.

---

## Setup & Reproducibility

To reproduce this analysis, ensure you have `pandas`, `seaborn`, and `scikit-learn` installed.

```python
import pandas as pd
import seaborn as sns
# (Full code provided in analysis.py/notebook)

```

**Submission Contact:** *Name: Prabhav Mishra* *Role: Data Science Intern Applicant*
