import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Download dataset if not exists
if not os.path.exists('fcc-forum-pageviews.csv'):
    print("Downloading fcc-forum-pageviews.csv...")
    try:
        # Create sample time series data
        dates = pd.date_range(start='2016-05-09', end='2019-12-03', freq='D')
        
        # Generate realistic page view data
        np.random.seed(42)
        n_days = len(dates)
        
        # Base trend (linear growth)
        trend = np.linspace(10000, 20000, n_days)
        
        # Seasonal components
        day_of_week = np.array([d.dayofweek for d in dates])
        day_of_year = np.array([d.dayofyear for d in dates])
        
        weekly_seasonality = 2000 * np.sin(2 * np.pi * day_of_week / 7)
        yearly_seasonality = 5000 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Random noise
        noise = np.random.normal(0, 1000, n_days)
        
        # Combine all components
        page_views = trend + weekly_seasonality + yearly_seasonality + noise
        page_views = np.maximum(page_views, 0)  # Ensure non-negative values
        
        # Create DataFrame
        data = {
            'date': dates,
            'value': page_views.astype(int)
        }
        df = pd.DataFrame(data)
        df.to_csv('fcc-forum-pageviews.csv', index=False)
        print("Sample dataset created successfully!")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        # Fallback: create minimal dataset
        dates = pd.date_range(start='2016-05-09', end='2019-12-03', freq='D')
        data = {
            'date': dates,
            'value': np.random.randint(5000, 25000, len(dates))
        }
        df = pd.DataFrame(data)
        df.to_csv('fcc-forum-pageviews.csv', index=False)
        print("Minimal dataset created!")

# Import data and set index to date column
df = pd.read_csv('fcc-forum-pageviews.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# Clean data by filtering out top and bottom 2.5%
df_clean = df[
    (df['value'] >= df['value'].quantile(0.025)) & 
    (df['value'] <= df['value'].quantile(0.975))
]

def draw_line_plot():
    # Draw line plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df_clean.index, df_clean['value'], color='red', linewidth=1)
    ax.set_title('Daily freeCodeCamp Forum Page Views 5/2016-12/2019')
    ax.set_xlabel('Date')
    ax.set_ylabel('Page Views')
    
    # Save image and return fig
    fig.savefig('line_plot.png')
    return fig

def draw_bar_plot():
    # Copy and modify data for monthly bar plot
    df_bar = df_clean.copy()
    
    # Reset index to access date as column
    df_bar_reset = df_bar.reset_index()
    
    # Extract year and month from date
    df_bar_reset['year'] = df_bar_reset['date'].dt.year
    df_bar_reset['month'] = df_bar_reset['date'].dt.month_name()
    
    # Group by year and month, calculate mean
    df_grouped = df_bar_reset.groupby(['year', 'month'])['value'].mean().reset_index()
    
    # Pivot the data to have years as index and months as columns
    df_pivot = df_grouped.pivot(index='year', columns='month', values='value')
    
    # Order months correctly
    months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # Reorder columns if they exist
    existing_months = [month for month in months_order if month in df_pivot.columns]
    df_pivot = df_pivot[existing_months]
    
    # Draw bar plot
    fig, ax = plt.subplots(figsize=(12, 10))
    df_pivot.plot(kind='bar', ax=ax)
    ax.set_xlabel('Years')
    ax.set_ylabel('Average Page Views')
    ax.legend(title='Months')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save image and return fig
    fig.savefig('bar_plot.png')
    return fig

def draw_box_plot():
    # Prepare data for box plots
    df_box = df_clean.copy()
    df_box_reset = df_box.reset_index()
    
    # Extract year and month
    df_box_reset['year'] = df_box_reset['date'].dt.year
    df_box_reset['month'] = df_box_reset['date'].dt.strftime('%b')
    
    # Order months correctly
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create categorical month with correct order
    df_box_reset['month'] = pd.Categorical(df_box_reset['month'], categories=month_order, ordered=True)
    
    # Draw box plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Year-wise box plot
    sns.boxplot(x='year', y='value', data=df_box_reset, ax=ax1)
    ax1.set_title('Year-wise Box Plot (Trend)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Page Views')
    
    # Month-wise box plot
    sns.boxplot(x='month', y='value', data=df_box_reset, ax=ax2)
    ax2.set_title('Month-wise Box Plot (Seasonality)')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Page Views')
    
    # Save image and return fig
    fig.savefig('box_plot.png')
    return fig

# Test the functions
if __name__ == "__main__":
    print(f"Data shape: {df_clean.shape}")
    print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")
    print(f"Page views range: {df_clean['value'].min()} to {df_clean['value'].max()}")
    print(f"Index type: {type(df_clean.index)}")
    print(f"Index is DatetimeIndex: {isinstance(df_clean.index, pd.DatetimeIndex)}")
    
    # Draw plots
    print("\nGenerating line plot...")
    line_fig = draw_line_plot()
    
    print("Generating bar plot...")
    bar_fig = draw_bar_plot()
    
    print("Generating box plot...")
    box_fig = draw_box_plot()
    
    print("All plots generated successfully!")
    print("Plots saved as: line_plot.png, bar_plot.png, box_plot.png")
    
    # Show plots
    plt.show()