import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# STEP 1: LOAD & CLEAN DATA 
sentiment_df = pd.read_csv('csvs/fear_greed_index.csv')
trades_df = pd.read_csv('csvs/historical_data.csv')

# Standardize date formats for merging
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
trades_df['date'] = pd.to_datetime(trades_df['Timestamp IST'], dayfirst=True).dt.normalize()

# Align datasets: Attach market sentiment to every individual trade execution
merged_df = pd.merge(trades_df, sentiment_df[['date', 'value', 'classification']], on='date', how='left')

# STEP 2: FEATURE ENGINEERING
# Define Win/Loss and directional bias flags
merged_df['is_win'] = (merged_df['Closed PnL'] > 0).astype(int)
merged_df['is_long'] = (merged_df['Side'].str.upper() == 'BUY').astype(int)
merged_df['is_short'] = (merged_df['Side'].str.upper() == 'SELL').astype(int)

# Aggregate data to Daily Performance per Trader (Account)
daily_metrics = merged_df.groupby(['Account', 'date', 'classification']).agg(
    daily_pnl=('Closed PnL', 'sum'),
    trade_count=('Trade ID', 'count'),
    total_size_usd=('Size USD', 'sum'),
    avg_trade_size=('Size USD', 'mean'),
    wins=('is_win', 'sum'),
    long_count=('is_long', 'sum'),
    short_count=('is_short', 'sum')
).reset_index()

# Calculate key ratios
daily_metrics['win_rate'] = daily_metrics['wins'] / daily_metrics['trade_count']
daily_metrics['ls_ratio'] = daily_metrics['long_count'] / (daily_metrics['short_count'] + 1e-9)

# STEP 3: SEGMENTATION 
# Segment traders into 'Frequent' and 'Infrequent' based on median activity
user_activity = daily_metrics.groupby('Account')['trade_count'].sum().reset_index()
user_activity['segment'] = pd.qcut(user_activity['trade_count'], 2, labels=['Infrequent', 'Frequent'])

# Merge segments back to the daily performance data
daily_metrics = pd.merge(daily_metrics, user_activity[['Account', 'segment']], on='Account')

# Set categorical order for better chart readability
sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
daily_metrics['classification'] = pd.Categorical(daily_metrics['classification'], categories=sentiment_order, ordered=True)

# STEP 4: GENERATE IMAGES
sns.set(style="whitegrid")

# 1. Win Rate Analysis
plt.figure(figsize=(10, 6))
wr_data = daily_metrics.groupby('classification')['win_rate'].mean()
sns.barplot(x=wr_data.index, y=wr_data.values, palette='RdYlGn')
plt.title('Average Win Rate across Market Sentiment')
plt.ylabel('Win Rate')
plt.savefig('win_rate_sentiment.png')

# 2. Behavior Boxplot (Frequency Distribution)
plt.figure(figsize=(12, 6))
sns.boxplot(data=daily_metrics, x='classification', y='trade_count', hue='segment')
plt.yscale('log') # Log scale helps visualize outliers in trading frequency
plt.title('Trade Frequency Distribution by Segment & Sentiment')
plt.ylabel('Daily Trade Count (Log Scale)')
plt.savefig('behavior_boxplot.png')

# 3. Segmented PnL Performance
plt.figure(figsize=(12, 6))
pnl_summary = daily_metrics.groupby(['classification', 'segment'])['daily_pnl'].mean().reset_index()
sns.barplot(data=pnl_summary, x='classification', y='daily_pnl', hue='segment', palette='coolwarm')
plt.title('Average Daily PnL by Trader Segment & Sentiment')
plt.ylabel('Avg Daily PnL ($)')
plt.savefig('segment_pnl_sentiment.png')

# STEP 5: PREDICTIVE MODEL (Bonus) 
X = daily_metrics[['trade_count', 'avg_trade_size', 'ls_ratio']].dropna()
y = (daily_metrics.loc[X.index, 'daily_pnl'] > 0).astype(int)

# Train a simple Random Forest to identify key drivers of success
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Capture Feature Importance
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_}).sort_values('Importance', ascending=False)

# STEP 6: EXPORT FINAL CSVs
merged_df.to_csv('processed_trades_sentiment.csv', index=False)
daily_metrics.to_csv('trader_daily_performance.csv', index=False)
user_activity.to_csv('trader_segments.csv', index=False)
feat_imp.to_csv('feature_importance.csv', index=False)

