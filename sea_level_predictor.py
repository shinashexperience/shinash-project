import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import os

def download_sea_level_data():
    """Download data sea level dari GitHub freeCodeCamp jika file tidak ada"""
    import urllib.request
    
    url = "https://raw.githubusercontent.com/freeCodeCamp/boilerplate-sea-level-predictor/master/epa-sea-level.csv"
    filename = "epa-sea-level.csv"
    
    if not os.path.exists(filename):
        print("Downloading data file...")
        urllib.request.urlretrieve(url, filename)
        print("Download completed!")
    else:
        print("Data file already exists.")

def sea_level_predictor():
    # Download data jika belum ada
    download_sea_level_data()
    
    # Baca data dari CSV
    try:
        df = pd.read_csv('epa-sea-level.csv')
        print("Data loaded successfully!")
        print(f"Data shape: {df.shape}")
        print(f"Years range: {df['Year'].min()} - {df['Year'].max()}")
    except FileNotFoundError:
        print("Error: File still not found after download attempt.")
        return None
    
    # Buat scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(df['Year'], df['CSIRO Adjusted Sea Level'], color='blue', alpha=0.7, label='Data Observasi')
    
    # 1. Garis regresi untuk seluruh data (1880-2013)
    slope, intercept, r_value, p_value, std_err = linregress(df['Year'], df['CSIRO Adjusted Sea Level'])
    
    # Extend tahun hingga 2050 untuk prediksi
    years_extended = np.arange(1880, 2051)
    sea_levels_predicted = intercept + slope * years_extended
    
    plt.plot(years_extended, sea_levels_predicted, 'r-', linewidth=2, 
             label=f'Regresi 1880-2013 (slope: {slope:.4f} in/tahun)')
    
    # 2. Garis regresi hanya untuk data sejak tahun 2000
    df_recent = df[df['Year'] >= 2000]
    
    if len(df_recent) > 1:  # Pastikan ada cukup data
        slope_recent, intercept_recent, r_value_recent, p_value_recent, std_err_recent = linregress(
            df_recent['Year'], df_recent['CSIRO Adjusted Sea Level'])
        
        years_recent_extended = np.arange(2000, 2051)
        sea_levels_recent_predicted = intercept_recent + slope_recent * years_recent_extended
        
        plt.plot(years_recent_extended, sea_levels_recent_predicted, 'g-', linewidth=2, 
                 label=f'Regresi 2000-2013 (slope: {slope_recent:.4f} in/tahun)')
        
        # Prediksi untuk tahun 2050
        prediction_2050_full = intercept + slope * 2050
        prediction_2050_recent = intercept_recent + slope_recent * 2050
        
        print(f"Prediksi kenaikan permukaan laut tahun 2050:")
        print(f"- Berdasarkan data 1880-2013: {prediction_2050_full:.2f} inches")
        print(f"- Berdasarkan data 2000-2013: {prediction_2050_recent:.2f} inches")
    
    # Konfigurasi plot
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Sea Level (inches)', fontsize=12)
    plt.title('Rise in Sea Level', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Simpan plot
    plt.savefig('sea_level_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'sea_level_plot.png'")
    
    return plt.gcf()

# Jalankan fungsi jika di-execute langsung
if __name__ == '__main__':
    sea_level_predictor()
    plt.show()