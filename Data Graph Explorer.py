import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import os
from urllib.parse import urlparse

class DataGraphExplorer:
    def __init__(self):
        self.df = None
        self.columns = []
        
    def get_csv_three_ways(self):
        """Mendapatkan file CSV melalui 3 cara berbeda"""
        print("=== DATA GRAPH EXPLORER ===")
        print("Pilih metode input CSV:")
        print("1. Upload dari komputer lokal")
        print("2. Input URL dari user")
        print("3. Gunakan URL yang sudah ada di kode")
        
        choice = input("Masukkan pilihan (1/2/3): ").strip()
        
        if choice == "1":
            return self.upload_local()
        elif choice == "2":
            return self.input_url()
        elif choice == "3":
            return self.hardcoded_url()
        else:
            print("Pilihan tidak valid!")
            return False
    
    def upload_local(self):
        """Upload file dari komputer lokal"""
        file_path = input("Masukkan path file CSV: ").strip()
        
        # Jika file tidak ditemukan, coba buat file contoh
        if not os.path.exists(file_path):
            print("File tidak ditemukan. Membuat file contoh...")
            self.create_sample_csv()
            file_path = "sample_data.csv"
        
        try:
            self.df = pd.read_csv(file_path)
            print(f"File berhasil dibaca: {file_path}")
            return True
        except Exception as e:
            print(f"Error membaca file: {e}")
            return False
    
    def input_url(self):
        """Input URL dari user"""
        url = input("Masukkan URL CSV: ").strip()
        
        # Validasi URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Baca CSV dari response content
            self.df = pd.read_csv(io.StringIO(response.text))
            print(f"CSV berhasil diunduh dari: {url}")
            return True
        except Exception as e:
            print(f"Error mengunduh dari URL: {e}")
            return False
    
    def hardcoded_url(self):
        """Gunakan URL yang sudah ada di kode"""
        # Contoh URL dataset publik
        sample_urls = {
            "1": "https://raw.githubusercontent.com/datasets/iris/master/data/iris.csv",
            "2": "https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv",
            "3": "https://raw.githubusercontent.com/datasets/world-cities/master/data/world-cities.csv"
        }
        
        print("\nPilih dataset contoh:")
        for key, url in sample_urls.items():
            print(f"{key}. {url.split('/')[-2]}")
        
        choice = input("Pilih dataset (1/2/3): ").strip()
        
        if choice in sample_urls:
            try:
                response = requests.get(sample_urls[choice])
                response.raise_for_status()
                self.df = pd.read_csv(io.StringIO(response.text))
                print(f"Dataset berhasil diunduh!")
                return True
            except Exception as e:
                print(f"Error mengunduh dataset: {e}")
                return False
        else:
            print("Pilihan tidak valid!")
            return False
    
    def create_sample_csv(self):
        """Membuat file CSV contoh jika file lokal tidak ditemukan"""
        sample_data = {
            'Tahun': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
            'Pendapatan': [100, 120, 150, 180, 200, 220, 250, 280, 300, 320],
            'Pengeluaran': [80, 90, 110, 130, 150, 160, 180, 200, 210, 220],
            'Profit': [20, 30, 40, 50, 50, 60, 70, 80, 90, 100]
        }
        
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv('sample_data.csv', index=False)
        print("File contoh 'sample_data.csv' telah dibuat!")
    
    def display_data_info(self):
        """Menampilkan informasi data"""
        if self.df is None:
            print("Data belum dimuat!")
            return
        
        print("\n=== INFORMASI DATA ===")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        print("\n2 Baris pertama:")
        print(self.df.head(2))
        
        print("\nInfo dataset:")
        print(self.df.info())
        
        # Simpan nama kolom sebagai list
        self.columns = list(self.df.columns)
        print(f"\nDaftar kolom: {self.columns}")
    
    def select_columns_and_plot(self):
        """Memilih kolom dan membuat plot"""
        if self.df is None:
            print("Data belum dimuat!")
            return
        
        print("\n=== PEMILIHAN KOLOM ===")
        print("Daftar kolom yang tersedia:")
        for i, col in enumerate(self.columns, 1):
            print(f"{i}. {col}")
        
        try:
            # Pilih kolom
            col_choice1 = input("Pilih kolom pertama (nomor): ").strip()
            col_choice2 = input("Pilih kolom kedua (opsional, kosongkan untuk line chart): ").strip()
            
            col1_idx = int(col_choice1) - 1
            col1 = self.columns[col1_idx]
            
            if col_choice2:
                col2_idx = int(col_choice2) - 1
                col2 = self.columns[col2_idx]
                columns = [col1, col2]
            else:
                columns = [col1]
                col2 = None
            
            # Konversi ke numpy array
            self.convert_to_numpy_and_plot(columns, col2 is not None)
            
        except (ValueError, IndexError) as e:
            print(f"Error memilih kolom: {e}")
    
    def convert_to_numpy_and_plot(self, columns, is_scatter=False):
        """Konversi ke numpy array dan buat plot"""
        try:
            # Konversi ke numpy arrays
            arrays = {}
            for col in columns:
                arrays[col] = self.df[col].to_numpy()
                print(f"\n{col} sebagai numpy array (5 nilai pertama):")
                print(arrays[col][:5])
            
            # Buat plot
            plt.figure(figsize=(10, 6))
            
            if is_scatter and len(columns) == 2:
                # Scatter plot untuk 2 kolom
                plt.scatter(arrays[columns[0]], arrays[columns[1]], alpha=0.7, color='blue')
                plt.xlabel(columns[0])
                plt.ylabel(columns[1])
                plt.title(f'Scatter Plot: {columns[0]} vs {columns[1]}')
                plt.grid(True, alpha=0.3)
                
                # Interpretasi
                correlation = np.corrcoef(arrays[columns[0]], arrays[columns[1]])[0,1]
                print(f"\n=== INTERPRETASI ===")
                print(f"Korelasi antara {columns[0]} dan {columns[1]}: {correlation:.3f}")
                if correlation > 0.7:
                    print("Korelasi positif kuat")
                elif correlation > 0.3:
                    print("Korelasi positif sedang")
                elif correlation > -0.3:
                    print("Korelasi lemah")
                elif correlation > -0.7:
                    print("Korelasi negatif sedang")
                else:
                    print("Korelasi negatif kuat")
                    
            else:
                # Line plot untuk 1 kolom atau multiple lines
                for col in columns:
                    plt.plot(arrays[col], label=col, marker='o')
                
                plt.xlabel('Index')
                plt.ylabel('Nilai')
                plt.title(f'Line Chart: {" vs ".join(columns)}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                print(f"\n=== INTERPRETASI ===")
                for col in columns:
                    print(f"{col}: Min={arrays[col].min():.2f}, Max={arrays[col].max():.2f}, Mean={arrays[col].mean():.2f}")
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error membuat plot: {e}")
    
    def run(self):
        """Menjalankan aplikasi utama"""
        # Dapatkan data CSV
        if not self.get_csv_three_ways():
            return
        
        # Tampilkan informasi data
        self.display_data_info()
        
        # Loop untuk multiple plots
        while True:
            self.select_columns_and_plot()
            
            continue_choice = input("\nApakah ingin mencoba kombinasi kolom lain? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("Terima kasih telah menggunakan Data Graph Explorer!")
                break

# Jalankan aplikasi
if __name__ == "__main__":
    explorer = DataGraphExplorer()
    explorer.run()