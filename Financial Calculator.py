import math

def calculate_annuity():
    """Menghitung nilai anuitas dengan pertumbuhan bulanan atau kontinu"""
    print("\n=== KALKULATOR ANUITAS ===")
    p = float(input("Nilai pokok (P): "))
    r = float(input("Tingkat bunga tahunan (dalam desimal, contoh 0.05 untuk 5%): "))
    t = float(input("Jangka waktu (tahun): "))
    
    print("Pilih jenis pertumbuhan:")
    print("1. Bulanan")
    print("2. Kontinu")
    choice = input("Pilihan (1/2): ")
    
    if choice == '1':
        n = 12  # compounding bulanan
        a = p * (1 + r/n)**(n*t)
        print(f"Nilai anuitas dengan pertumbuhan bulanan: ${a:,.2f}")
    elif choice == '2':
        a = p * math.exp(r*t)  # pertumbuhan kontinu
        print(f"Nilai anuitas dengan pertumbuhan kontinu: ${a:,.2f}")
    else:
        print("Pilihan tidak valid")

def calculate_mortgage():
    """Menghitung pembayaran hipotek bulanan"""
    print("\n=== KALKULATOR HIPOTEK ===")
    principal = float(input("Jumlah pinjaman: "))
    annual_rate = float(input("Tingkat bunga tahunan (dalam desimal): "))
    years = int(input("Jangka waktu (tahun): "))
    
    monthly_rate = annual_rate / 12
    months = years * 12
    
    # Rumus pembayaran hipotek bulanan
    monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
    
    print(f"Pembayaran bulanan: ${monthly_payment:,.2f}")
    print(f"Total pembayaran: ${monthly_payment * months:,.2f}")
    print(f"Total bunga: ${monthly_payment * months - principal:,.2f}")

def estimate_retirement():
    """Memperkirakan saldo investasi pensiun"""
    print("\n=== ESTIMASI INVESTASI PENSIUN ===")
    initial = float(input("Investasi awal: "))
    monthly_contribution = float(input("Kontribusi bulanan: "))
    annual_return = float(input("Pengembalian tahunan yang diharapkan (dalam desimal): "))
    years = int(input("Jumlah tahun hingga pensiun: "))
    
    monthly_return = annual_return / 12
    months = years * 12
    
    # Future value dengan kontribusi bulanan
    fv = initial * (1 + monthly_return)**months
    fv += monthly_contribution * (((1 + monthly_return)**months - 1) / monthly_return)
    
    print(f"Perkiraan saldo pensiun: ${fv:,.2f}")
    total_contributions = initial + (monthly_contribution * months)
    print(f"Total kontribusi: ${total_contributions:,.2f}")
    print(f"Perkiraan pertumbuhan: ${fv - total_contributions:,.2f}")

def doubling_time():
    """Menentukan waktu yang dibutuhkan untuk melipatgandakan investasi"""
    print("\n=== WAKTU PELIPATGANDAAN ===")
    r = float(input("Tingkat bunga tahunan (dalam desimal): "))
    
    # Aturan 72 (approximasi)
    time_rule72 = 72 / (r * 100)
    
    # Perhitungan eksak
    time_exact = math.log(2) / math.log(1 + r)
    
    print(f"Waktu pelipatgandaan (Aturan 72): {time_rule72:.2f} tahun")
    print(f"Waktu pelipatgandaan (perhitungan eksak): {time_exact:.2f} tahun")

def solve_logarithmic():
    """Menyelesaikan persamaan logaritmik"""
    print("\n=== SOLVER PERSAMAAN LOGARITMIK ===")
    print("Format persamaan: a * b^x = c")
    a = float(input("Nilai a: "))
    b = float(input("Nilai b: "))
    c = float(input("Nilai c: "))
    
    if a <= 0 or b <= 0 or c <= 0:
        print("Error: Nilai harus positif")
        return
    
    x = math.log(c / a) / math.log(b)
    print(f"Nilai x adalah: {x:.4f}")
    print(f"Verifikasi: {a} * {b}^{x:.4f} = {a * (b ** x):.4f}")

def scientific_notation_converter():
    """Mengkonversi ke dan dari notasi ilmiah"""
    print("\n=== KONVERTER NOTASI ILMIAH ===")
    print("1. Ke notasi ilmiah")
    print("2. Dari notasi ilmiah")
    choice = input("Pilihan (1/2): ")
    
    if choice == '1':
        number = float(input("Masukkan angka: "))
        print(f"Notasi ilmiah: {number:.2e}")
    elif choice == '2':
        sci_notation = input("Masukkan notasi ilmiah (contoh: 1.23e4): ")
        try:
            number = float(sci_notation)
            print(f"Bentuk standar: {number:,.2f}")
        except ValueError:
            print("Format notasi ilmiah tidak valid")
    else:
        print("Pilihan tidak valid")

def main():
    """Menu utama kalkulator finansial"""
    while True:
        print("\n" + "="*50)
        print("KALKULATOR FINANSIAL")
        print("="*50)
        print("1. Hitung Anuitas")
        print("2. Hitung Pembayaran Hipotek")
        print("3. Estimasi Investasi Pensiun")
        print("4. Waktu Pelipatgandaan Investasi")
        print("5. Selesaikan Persamaan Logaritmik")
        print("6. Konverter Notasi Ilmiah")
        print("7. Keluar")
        
        choice = input("Pilih menu (1-7): ")
        
        if choice == '1':
            calculate_annuity()
        elif choice == '2':
            calculate_mortgage()
        elif choice == '3':
            estimate_retirement()
        elif choice == '4':
            doubling_time()
        elif choice == '5':
            solve_logarithmic()
        elif choice == '6':
            scientific_notation_converter()
        elif choice == '7':
            print("Terima kasih telah menggunakan kalkulator finansial!")
            break
        else:
            print("Pilihan tidak valid. Silakan pilih 1-7.")
        
        input("\nTekan Enter untuk melanjutkan...")

if __name__ == "__main__":
    main()