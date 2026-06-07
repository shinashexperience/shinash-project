# CAT Kalkulus 1A — Bab 1: Sistem Bilangan Real

**Computer Assisted Test (CAT) berbasis GUI untuk mata kuliah MAA1114**  
S1 Teknik Elektro — Institut Teknologi Telkom Surabaya

---

## Deskripsi

**CAT Kalkulus 1A** adalah aplikasi ujian berbasis komputer (*Computer Assisted Test*) yang dirancang khusus untuk mendukung evaluasi mandiri mahasiswa pada mata kuliah **Kalkulus 1A (MAA1114)**, Bab 1: Sistem Bilangan Real. Aplikasi ini dibangun menggunakan Python dan antarmuka grafis Tkinter, sehingga dapat dijalankan di seluruh platform tanpa memerlukan instalasi library tambahan di luar distribusi Python standar.

Aplikasi mensimulasikan suasana ujian resmi dengan dukungan timer, navigasi soal, sistem tandai, dan pembahasan lengkap di akhir sesi — semuanya dalam antarmuka bertema gelap (*dark mode*) yang modern.

---

## Fitur Utama

| Fitur | Keterangan |
|---|---|
| **20 Soal Pilihan Ganda** | Tersebar dalam 4 bagian tematik (A, B, C, D) |
| **Timer 40 Menit** | Hitung mundur real-time; peringatan visual di 10 & 5 menit terakhir |
| **Navigasi Soal** | Sidebar dengan indikator status: belum dijawab, dijawab, ditandai, aktif |
| **Sistem Tandai** | Soal dapat ditandai (flag) untuk ditinjau ulang sebelum dikumpulkan |
| **Konfirmasi Pengumpulan** | Ringkasan status jawaban sebelum ujian diselesaikan |
| **Penilaian Otomatis** | Skor, grade (A–E), dan statistik per bagian dihitung seketika |
| **Pembahasan Detail** | Penjelasan per soal dapat ditampilkan/disembunyikan secara interaktif |
| **Konteks Teknik Elektro** | Soal aplikatif melibatkan skenario RC circuit, Op-Amp, dan toleransi resistor |
| **Dark Mode UI** | Antarmuka modern bertema gelap dengan palet warna yang konsisten |

---

## Persyaratan Sistem

| Komponen | Minimum |
|---|---|
| **Python** | 3.7 atau lebih baru |
| **Library** | Hanya Tkinter (sudah termasuk dalam Python standar) |
| **OS** | Windows 10+, macOS 10.14+, Linux (dengan Tkinter tersedia) |
| **Resolusi Layar** | Minimum 900 × 600 piksel (direkomendasikan 1100 × 720) |

> **Catatan untuk pengguna Linux:** Jika Tkinter belum tersedia, jalankan:  
> `sudo apt-get install python3-tk` (Debian/Ubuntu)  
> `sudo dnf install python3-tkinter` (Fedora/RHEL)

---

## Cara Menjalankan

### 1. Verifikasi Python

```bash
python --version
# atau
python3 --version
```

Pastikan versi Python yang digunakan adalah 3.7 ke atas.

### 2. Jalankan Aplikasi

```bash
python CATV2.py
# atau
python3 CATV2.py
```

Tidak diperlukan langkah instalasi dependensi tambahan.

---

## Alur Penggunaan

```
[Layar Login] → Isi Nama, NIM, Kelas → [Mulai Ujian]
     ↓
[Layar Ujian] → Jawab soal, navigasi, tandai soal → [Selesaikan Ujian]
     ↓
[Layar Rekap] → Lihat nilai, statistik, dan pembahasan → [Ulangi / Keluar]
```

### Langkah Detail

1. **Layar Login**  
   Isi nama lengkap, NIM, dan pilih kelas (TE-A hingga TE-E atau Lainnya). Klik **MULAI UJIAN** atau tekan `Enter`.

2. **Layar Ujian**  
   - Klik opsi pilihan ganda untuk menjawab soal.
   - Gunakan sidebar kiri untuk melompat ke soal manapun.
   - Klik **Tandai Soal** untuk menandai soal yang ingin ditinjau ulang.
   - Soal terakhir menampilkan tombol **Selesaikan ✓** sebagai ganti tombol Berikutnya.
   - Timer berwarna kuning saat ≤ 10 menit tersisa, merah saat ≤ 5 menit tersisa.
   - Jika waktu habis, ujian diselesaikan secara otomatis.

3. **Layar Rekap**  
   - Nilai akhir ditampilkan dalam format 0–100 beserta grade huruf (A, B, C, D, atau E).
   - Statistik per bagian tersedia dalam bentuk kartu ringkasan.
   - Setiap soal dapat diekspansi untuk melihat kunci jawaban dan pembahasan lengkap.
   - Pilih **Ulangi Ujian** untuk kembali ke layar login atau **Keluar** untuk menutup aplikasi.

---

## Struktur Soal

| Bagian | Topik | Jumlah Soal | Poin per Soal |
|---|---|---|---|
| **A** | Himpunan Bilangan | 5 | 5 |
| **B** | Nilai Mutlak | 5 | 5 |
| **C** | Pertidaksamaan | 5 | 5 |
| **D** | Aplikasi & UTS | 5 | 5 |
| **Total** | | **20** | **100** |

---

## Sistem Penilaian

| Grade | Rentang Nilai | Keterangan |
|---|---|---|
| **A** | ≥ 90 | Luar biasa |
| **B** | 80 – 89 | Baik |
| **C** | 70 – 79 | Cukup |
| **D** | 60 – 69 | Kurang |
| **E** | < 60 | Tidak Lulus |

Rumus nilai: `Nilai = (Poin Diperoleh / 100) × 100`

---

## Indikator Warna Navigasi

| Warna | Status |
|---|---|
| 🔵 Biru (per Bagian) | Soal aktif / sedang ditampilkan |
| 🟢 Hijau | Soal sudah dijawab |
| 🟠 Oranye | Soal ditandai (flag) |
| Abu-abu Gelap | Soal belum dijawab |

---

## Struktur File

```
CATV2.py          ← Seluruh source code aplikasi (single-file)
README.md         ← Dokumen ini
DOKUMENTASI.md    ← Dokumentasi teknis lengkap
```

---

## Informasi Akademik

| Atribut | Detail |
|---|---|
| **Mata Kuliah** | Kalkulus 1A |
| **Kode MK** | MAA1114 |
| **Materi** | Bab 1: Sistem Bilangan Real |
| **Program Studi** | S1 Teknik Elektro |
| **Institusi** | Institut Teknologi Telkom Surabaya |

---

## Keterbatasan

- Hasil ujian tidak disimpan secara permanen ke file atau database.
- Tidak terdapat fitur enkripsi atau pengamanan soal untuk lingkungan ujian formal.
- Bank soal bersifat statis (tidak diacak antar sesi secara default).
- Tidak mendukung koneksi jaringan atau sistem LMS.

---

## Lisensi

Aplikasi ini dikembangkan untuk keperluan belajar mandiri mahasiswa Institut Teknologi Telkom Surabaya. Seluruh konten soal disesuaikan dengan Rencana Pembelajaran Semester (RPS) mata kuliah MAA1114.
