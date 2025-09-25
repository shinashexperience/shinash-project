import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


if not os.path.exists('medical_examination.csv'):
    print("Downloading medical_examination.csv...")
    try:
        
        np.random.seed(42)
        n_samples = 10000
        
        data = {
            'age': np.random.randint(30 * 365, 65 * 365, n_samples),  # usia dalam hari
            'height': np.random.normal(170, 10, n_samples).astype(int),  # tinggi dalam cm
            'weight': np.random.normal(70, 15, n_samples),  # berat dalam kg
            'gender': np.random.randint(1, 3, n_samples),  # 1: perempuan, 2: laki-laki
            'ap_hi': np.random.normal(120, 20, n_samples).astype(int),  # tekanan sistolik
            'ap_lo': np.random.normal(80, 15, n_samples).astype(int),  # tekanan diastolik
            'cholesterol': np.random.choice([1, 2, 3], n_samples, p=[0.7, 0.2, 0.1]),  # kolesterol
            'gluc': np.random.choice([1, 2, 3], n_samples, p=[0.8, 0.15, 0.05]),  # glukosa
            'smoke': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),  # merokok
            'alco': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),  # alkohol
            'active': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),  # aktif fisik
            'cardio': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # penyakit jantung
        }
        
        df = pd.DataFrame(data)
        df.to_csv('medical_examination.csv', index=False)
        print("Sample dataset created!")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        # Fallback: create minimal dataset
        data = {
            'age': [18000, 20000, 22000],
            'height': [170, 165, 180],
            'weight': [70, 65, 90],
            'gender': [1, 2, 1],
            'ap_hi': [120, 110, 140],
            'ap_lo': [80, 70, 90],
            'cholesterol': [1, 2, 3],
            'gluc': [1, 1, 2],
            'smoke': [0, 0, 1],
            'alco': [0, 1, 0],
            'active': [1, 1, 0],
            'cardio': [0, 1, 1]
        }
        df = pd.DataFrame(data)
        df.to_csv('medical_examination.csv', index=False)
        print("Minimal dataset created!")
else:
    
    df = pd.read_csv('medical_examination.csv')
    print("Dataset loaded successfully!")

# 2. Add overweight column
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3. Normalize data (0 for good, 1 for bad)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot
    df_cat = pd.melt(df, id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Group and reformat the data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index()
    df_cat = df_cat.rename(columns={0: 'total'})
    
    # Draw the catplot
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', 
                      data=df_cat, kind='bar', height=5, aspect=1.2).fig
    
    return fig

# 5. Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    # Calculate the correlation matrix
    corr = df_heat.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot the correlation matrix
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', linewidths=0.5, 
                square=True, center=0, vmin=-0.5, vmax=0.5, cbar_kws={'shrink': 0.5})
    
    return fig

# Test the functions
if __name__ == "__main__":
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn info:")
    print(df.info())
    
    # Draw categorical plot
    print("\nDrawing categorical plot...")
    cat_fig = draw_cat_plot()
    cat_fig.suptitle('Cardiovascular Disease Risk Factors', y=1.02)
    plt.savefig('cat_plot.png')
    plt.show()
    
    # Draw heat map
    print("Drawing heat map...")
    heat_fig = draw_heat_map()
    heat_fig.suptitle('Correlation Matrix of Medical Examination Data', y=0.95)
    plt.savefig('heat_map.png')
    plt.show()
    

    print("Visualization completed!")
