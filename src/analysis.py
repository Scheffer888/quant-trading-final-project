import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

def analysis(feats_df, timeframe, lag, threshold):
    """
    Some Analysis on data focusing on effects of size_imbalance on price.
    """
    feats_df['lagged'] = feats_df['size_imbalance'].shift(lag)
    feats_df = feats_df.dropna()
    start_time = feats_df['time_trade'].min()
    end_time = start_time + pd.Timedelta(minutes=timeframe)
    df_subset = feats_df[(feats_df['time_trade'] >= start_time) & (feats_df['time_trade'] <= end_time)]
    
    plt.figure(figsize=(12, 6))

    # Plot price over time
    plt.plot(df_subset['time_trade'], df_subset['mid_price'], color='black', linewidth=1, label='Price')

    # Scatter plot with different colors for positive and negative SI
    plt.scatter(df_subset['time_trade'][df_subset['lagged'] > 0], df_subset['mid_price'][df_subset['lagged'] > 0], 
                color='green', label='Positive SI', alpha=0.6, s=20)
    plt.scatter(df_subset['time_trade'][df_subset['lagged'] < 0], df_subset['mid_price'][df_subset['lagged'] < 0], 
                color='red', label='Negative SI', alpha=0.6, s=20)

    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"Price Movement with SI labels ({start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    feats_df['log_return'] = np.log(feats_df['mid_price']).diff()
    feats_df['price_return'] = feats_df['mid_price'].diff()
    feats_df = feats_df.dropna()

    corr, p_value = pearsonr(feats_df['lagged'], feats_df['ewma_price_return'])

    print(f"Correlation between SI and price change: {corr:.4f}")
    print(f"P-value: {p_value:.4e}")

    if p_value < 0.05:
        print("The correlation is statistically significant.")
    else:
        print("The correlation is not statistically significant.")

    significant_changes = feats_df[abs(feats_df['ewma_price_return']) > threshold]
    mismatched_signs = ((significant_changes['size_imbalance'] > 0) & (significant_changes['ewma_price_return'] < 0)) | \
                   ((significant_changes['size_imbalance'] < 0) & (significant_changes['ewma_price_return'] > 0))

    # Compute percentage
    percent_mismatch = 100- (mismatched_signs.sum() / len(significant_changes)) * 100

    print(f"Percentage of time SI and significant price change had same signs: {percent_mismatch:.2f}%")
    return