# Dokumentasi Teknis — CATV2.py

**CAT Kalkulus 1A | Versi 2.0 | GUI Edition**  
MAA1114 — S1 Teknik Elektro, Institut Teknologi Telkom Surabaya

---

## Daftar Isi

1. [Gambaran Umum Arsitektur](#1-gambaran-umum-arsitektur)
2. [Dependensi dan Lingkungan](#2-dependensi-dan-lingkungan)
3. [Struktur Kode](#3-struktur-kode)
4. [Model Data](#4-model-data)
5. [Konfigurasi Global](#5-konfigurasi-global)
6. [Sistem Warna](#6-sistem-warna)
7. [Komponen Kelas](#7-komponen-kelas)
   - [LayarLogin](#71-layarlogin)
   - [LayarUjian](#72-layarujian)
   - [LayarRekap](#73-layarrekap)
   - [AplikasiCAT](#74-aplikasicat)
8. [Alur Data dan State Management](#8-alur-data-dan-state-management)
9. [Logika Penilaian](#9-logika-penilaian)
10. [Sistem Timer](#10-sistem-timer)
11. [Bank Soal — Ringkasan Konten](#11-bank-soal--ringkasan-konten)
12. [Diagram Alur Aplikasi](#12-diagram-alur-aplikasi)
13. [Catatan Pengembangan](#13-catatan-pengembangan)

---

## 1. Gambaran Umum Arsitektur

`CATV2.py` adalah aplikasi **single-file** yang dibangun dengan paradigma **OOP (Object-Oriented Programming)** menggunakan framework GUI **Tkinter**. Arsitektur keseluruhan mengikuti pola **Frame-Based Navigation**, di mana setiap "layar" merupakan subkelas `tk.Frame` yang diinstansiasi dan dihancurkan oleh kelas pengontrol pusat (`AplikasiCAT`).

```
AplikasiCAT (Pengontrol Pusat)
├── LayarLogin      ← Layar 1: Identitas peserta
├── LayarUjian      ← Layar 2: Sesi ujian aktif
└── LayarRekap      ← Layar 3: Rekap hasil & pembahasan
```

**Komunikasi antar layar** dilakukan melalui callback (`on_mulai`, `on_selesai`, `on_ulang`) yang diteruskan sebagai parameter konstruktor, sehingga setiap layar tetap terpisah secara logika namun terhubung melalui kontrak antarmuka yang jelas.

---

## 2. Dependensi dan Lingkungan

### Library yang Digunakan

| Library | Modul | Kegunaan |
|---|---|---|
| `tkinter` | `tk` | Widget dasar GUI, window management |
| `tkinter` | `ttk` | Widget bertema (Combobox) |
| `tkinter` | `messagebox` | Dialog peringatan dan konfirmasi |
| `tkinter` | `font` | Pengelolaan font (diimport, reserved) |
| `time` | — | Pengelolaan waktu (diimport, reserved) |
| `random` | — | Pengacakan (diimport, reserved) |
| `dataclasses` | `dataclass`, `field` | Definisi model data `Soal` |
| `typing` | `Optional` | Anotasi tipe |
| `math` | — | Fungsi matematika (diimport, reserved) |

> Seluruh library di atas merupakan bagian dari **Python Standard Library** dan tidak memerlukan instalasi tambahan via `pip`.

### Versi Python yang Didukung

- Python 3.7+ (karena penggunaan `dataclass` dan f-string)
- Direkomendasikan: Python 3.9+

---

## 3. Struktur Kode

File `CATV2.py` terorganisir dalam blok-blok yang dibatasi komentar visual untuk kemudahan navigasi:

```
CATV2.py
│
├── [Baris   1–13]  Docstring aplikasi
├── [Baris  15–20]  Import library
│
├── [Baris  22–32]  Definisi dataclass Soal
├── [Baris  34–415] BANK_SOAL — 20 objek Soal
│
├── [Baris 417–422] Konstanta konfigurasi ujian
├── [Baris 424–458] Kamus warna CLR
│
├── [Baris 460–562] class LayarLogin
├── [Baris 564–870] class LayarUjian
├── [Baris 872–1180] class LayarRekap
├── [Baris 1182–1230] class AplikasiCAT
│
└── [Baris 1232–1234] Entry point (__main__)
```

---

## 4. Model Data

### Kelas `Soal`

Didefinisikan sebagai Python `dataclass`, merepresentasikan satu soal ujian.

```python
@dataclass
class Soal:
    id:          int        # Nomor soal unik (1–20)
    bagian:      str        # Kelompok soal: "A", "B", "C", atau "D"
    topik:       str        # Label topik spesifik
    pertanyaan:  str        # Teks pertanyaan lengkap
    pilihan:     list       # Daftar 4 pilihan: ["A. ...", "B. ...", ...]
    kunci:       str        # Jawaban benar: "A", "B", "C", atau "D"
    penjelasan:  str        # Pembahasan lengkap step-by-step
    poin:        int = 5    # Bobot poin (default: 5)
```

### Koleksi `BANK_SOAL`

`BANK_SOAL` adalah `list` berisi 20 objek `Soal`, dibagi dalam 4 bagian masing-masing 5 soal:

| Bagian | Rentang `id` | Topik Utama |
|---|---|---|
| A | 1–5 | Klasifikasi bilangan, hierarki himpunan, notasi interval |
| B | 6–10 | Persamaan, pertidaksamaan, sifat nilai mutlak, aplikasi resistor |
| C | 11–15 | Pertidaksamaan linear, kuadrat, rasional, aksioma urutan |
| D | 16–20 | Op-Amp, transien RC, Aksioma Kelengkapan, operasi interval |

---

## 5. Konfigurasi Global

Konstanta berikut dapat dimodifikasi untuk menyesuaikan parameter ujian:

```python
DURASI_UJIAN_DETIK = 40 * 60   # Durasi ujian dalam detik (default: 2400 = 40 menit)
NAMA_UJIAN  = "Kalkulus 1A — Bab 1: Sistem Bilangan Real"
KODE_MK     = "MAA1114"
PRODI       = "S1 Teknik Elektro"
INSTITUSI   = "Institut Teknologi Telkom Surabaya"
```

---

## 6. Sistem Warna

Seluruh warna aplikasi dipusatkan dalam kamus `CLR`, memudahkan perubahan tema secara seragam:

```python
CLR = {
    # Latar & Panel
    "bg_dark":     "#0F1117",   # Latar belakang utama (sangat gelap)
    "bg_panel":    "#1A1D2E",   # Panel sidebar dan header
    "bg_card":     "#1E2235",   # Kartu soal dan konten
    "bg_input":    "#252840",   # Latar pilihan jawaban
    "bg_hover":    "#2D3154",   # State hover / jawaban dipilih

    # Aksen Fungsional
    "accent":      "#4C6EF5",   # Biru — elemen utama, soal aktif (Bagian A)
    "accent_glow": "#5C7CFA",   # Biru terang — hover tombol utama
    "success":     "#40C057",   # Hijau — soal benar / dijawab
    "warning":     "#FD7E14",   # Oranye — soal ditandai / timer 10 menit
    "danger":      "#FA5252",   # Merah — soal salah / timer 5 menit
    "gold":        "#FFD43B",   # Emas — poin, timer normal, Bagian D
    "teal":        "#20C997",   # Teal — Bagian B
    "purple":      "#9775FA",   # Ungu — Bagian C

    # Tipografi
    "text_bright": "#FFFFFF",
    "text_main":   "#CDD3F0",
    "text_dim":    "#7A84B3",
    "text_muted":  "#4A5280",

    # Border
    "border":      "#2D3154",
    "border_glow": "#4C6EF5",
}
```

**Pemetaan warna per bagian soal:**

| Bagian | Warna | Hex |
|---|---|---|
| A | `accent` (biru) | `#4C6EF5` |
| B | `teal` (hijau-biru) | `#20C997` |
| C | `purple` (ungu) | `#9775FA` |
| D | `gold` (emas) | `#FFD43B` |

---

## 7. Komponen Kelas

### 7.1 `LayarLogin`

**Warisan:** `tk.Frame`  
**Berkas logika:** Validasi input identitas peserta

#### Konstruktor

```python
LayarLogin(master: tk.Tk, on_mulai: Callable[[str, str, str], None])
```

| Parameter | Tipe | Keterangan |
|---|---|---|
| `master` | `tk.Tk` | Window induk |
| `on_mulai` | `Callable` | Callback dipanggil saat login berhasil; menerima `(nama, nim, kelas)` |

#### Elemen UI

| Widget | Jenis | Keterangan |
|---|---|---|
| `entry_nama` | `tk.Entry` | Input nama lengkap; wajib diisi |
| `entry_nim` | `tk.Entry` | Input NIM; wajib diisi |
| `combo_kelas` | `ttk.Combobox` | Pilihan kelas TE-A hingga TE-E, Lainnya |
| Tombol Mulai | `tk.Button` | Memanggil `_validasi_dan_mulai()` |

#### Metode

| Metode | Akses | Keterangan |
|---|---|---|
| `_bangun()` | Private | Membangun seluruh struktur widget layar login |
| `_validasi_dan_mulai()` | Private | Memvalidasi input dan memanggil `on_mulai` |

#### Binding Keyboard

- `<Return>` pada `entry_nama` → fokus pindah ke `entry_nim`
- `<Return>` pada `entry_nim` → memanggil `_validasi_dan_mulai()`

---

### 7.2 `LayarUjian`

**Warisan:** `tk.Frame`  
**Berkas logika:** Manajemen sesi ujian aktif, timer, navigasi, dan penyimpanan jawaban

#### Konstruktor

```python
LayarUjian(
    master: tk.Tk,
    nama: str,
    nim: str,
    kelas: str,
    on_selesai: Callable[[str, str, str, dict, set, int], None]
)
```

| Parameter | Tipe | Keterangan |
|---|---|---|
| `nama` | `str` | Nama peserta dari layar login |
| `nim` | `str` | NIM peserta |
| `kelas` | `str` | Kelas peserta |
| `on_selesai` | `Callable` | Callback saat ujian selesai; menerima `(nama, nim, kelas, jawaban, ditandai, waktu_detik)` |

#### State Internal

| Atribut | Tipe | Keterangan |
|---|---|---|
| `soal_list` | `list[Soal]` | Salinan `BANK_SOAL` untuk sesi aktif |
| `total` | `int` | Jumlah total soal (20) |
| `idx_aktif` | `int` | Indeks soal yang sedang ditampilkan |
| `jawaban` | `dict[int, str]` | Peta `{soal_id: huruf_jawaban}` |
| `ditandai` | `set[int]` | Kumpulan `soal_id` yang ditandai |
| `sisa_detik` | `int` | Sisa waktu ujian dalam detik |
| `selesai` | `bool` | Flag apakah ujian sudah diselesaikan |
| `timer_job` | `str\|None` | ID `after()` job untuk membatalkan timer |

#### Metode

| Metode | Akses | Keterangan |
|---|---|---|
| `_bangun()` | Private | Membangun struktur UI: header, sidebar, area soal |
| `_tampilkan_soal(idx)` | Private | Memuat dan merender soal ke-`idx` |
| `_simpan_jawaban()` | Private | Menyimpan pilihan ke `self.jawaban` dan memperbarui tampilan |
| `_toggle_tandai()` | Private | Menambah/menghapus soal dari `self.ditandai` |
| `_soal_prev()` | Private | Navigasi ke soal sebelumnya |
| `_soal_next()` | Private | Navigasi ke soal berikutnya |
| `_update_nav()` | Private | Memperbarui warna semua tombol navigasi sidebar |
| `_tick_timer()` | Private | Callback rekursif timer setiap 1 detik |
| `_konfirmasi_selesai()` | Private | Menampilkan dialog konfirmasi pengumpulan |
| `_selesaikan()` | Private | Menghentikan timer dan memanggil `on_selesai` |

#### Konstanta Kelas

```python
WARNA_BAGIAN = {
    "A": CLR["accent"],   # Biru
    "B": CLR["teal"],     # Teal
    "C": CLR["purple"],   # Ungu
    "D": CLR["gold"],     # Emas
}
```

#### Perilaku Timer

```
sisa_detik > 600    → Timer berwarna emas (normal)
600 ≥ sisa_detik > 300 → Timer berwarna oranye (peringatan)
sisa_detik ≤ 300    → Timer berwarna merah (kritis)
sisa_detik == 0     → Dialog "Waktu Habis" → _selesaikan() otomatis
```

---

### 7.3 `LayarRekap`

**Warisan:** `tk.Frame`  
**Berkas logika:** Penilaian otomatis, tampilan hasil, statistik per bagian, pembahasan soal

#### Konstruktor

```python
LayarRekap(
    master: tk.Tk,
    nama: str,
    nim: str,
    kelas: str,
    jawaban: dict,
    ditandai: set,
    waktu_detik: int,
    on_ulang: Callable
)
```

#### State Internal

| Atribut | Tipe | Keterangan |
|---|---|---|
| `benar` | `int` | Jumlah soal yang dijawab benar |
| `salah` | `int` | Jumlah soal yang dijawab salah |
| `kosong` | `int` | Jumlah soal yang tidak dijawab |
| `poin_total` | `int` | Total poin yang diperoleh |
| `nilai` | `float` | Nilai akhir (skala 0–100) |
| `grade` | `str` | Huruf grade: "A", "B", "C", "D", atau "E" |
| `grade_color` | `str` | Warna hex sesuai grade |
| `detail` | `list[tuple]` | Daftar `(soal, jawab, status)` untuk setiap soal |

#### Metode

| Metode | Akses | Keterangan |
|---|---|---|
| `_hitung_nilai()` | Private | Menghitung skor, grade, dan membangun `self.detail` |
| `_bangun()` | Private | Membangun container scroll utama |
| `_bangun_konten()` | Private | Merender semua elemen konten dalam scroll frame |
| `_buat_kartu_soal(parent, soal, jawab, status)` | Private | Membuat kartu pembahasan untuk satu soal |

#### Logika Penilaian (dalam `_hitung_nilai`)

```python
untuk setiap soal dalam BANK_SOAL:
    jika soal.id tidak ada dalam jawaban:
        status = "kosong"    → kosong += 1
    jika jawaban[soal.id] == soal.kunci:
        status = "benar"     → benar += 1; poin_total += soal.poin
    lainnya:
        status = "salah"     → salah += 1

nilai = (poin_total / total_poin_max) × 100
```

#### Kode Warna Pembahasan Soal

| Kondisi | Warna Latar | Ikon |
|---|---|---|
| Kunci & Dipilih (benar) | `success` hijau | ✅ |
| Kunci & Tidak Dipilih | `#1a3a1a` hijau gelap | ✔ |
| Dipilih & Bukan Kunci | `#3a1a1a` merah gelap | ✗ |
| Tidak Dipilih | `bg_card` | — |

---

### 7.4 `AplikasiCAT`

**Berkas logika:** Inisialisasi window utama dan manajemen navigasi antar layar

#### Konstruktor

```python
AplikasiCAT()
```

Menginisialisasi `tk.Tk`, menetapkan properti window, dan memulai dengan `_ke_login()`.

#### Properti Window

| Properti | Nilai |
|---|---|
| Judul | `"CAT — {NAMA_UJIAN} \| {INSTITUSI}"` |
| Ukuran default | `1100 × 720` piksel |
| Ukuran minimum | `900 × 600` piksel |
| Warna latar | `CLR["bg_dark"]` |

#### Metode

| Metode | Keterangan |
|---|---|
| `_bersihkan()` | Menghancurkan layar aktif saat ini |
| `_ke_login()` | Transisi ke `LayarLogin` |
| `_ke_ujian(nama, nim, kelas)` | Transisi ke `LayarUjian` dengan data peserta |
| `_ke_rekap(nama, nim, kelas, jawaban, ditandai, waktu_detik)` | Transisi ke `LayarRekap` dengan hasil ujian |
| `jalankan()` | Memulai `mainloop()` Tkinter |

---

## 8. Alur Data dan State Management

```
[LayarLogin]
    ├── Input: nama (str), nim (str), kelas (str)
    └── Output via on_mulai(nama, nim, kelas)
            ↓
[LayarUjian]
    ├── State: jawaban {soal_id: huruf}, ditandai {soal_id}
    ├── Input: nama, nim, kelas dari login
    └── Output via on_selesai(nama, nim, kelas, jawaban, ditandai, waktu_detik)
            ↓
[LayarRekap]
    ├── Input: seluruh state dari ujian
    └── Output via on_ulang() → kembali ke LayarLogin
```

Tidak ada mekanisme penyimpanan persisten (file, database). Seluruh state hidup dalam memori selama sesi berlangsung dan hilang saat aplikasi ditutup atau ujian diulang.

---

## 9. Logika Penilaian

### Perhitungan Nilai

```
Total Poin Maksimum = 20 soal × 5 poin = 100
Poin Diperoleh      = Σ(poin soal untuk setiap jawaban benar)
Nilai               = (Poin Diperoleh / 100) × 100
```

### Konversi Grade

```python
nilai >= 90  → grade = "A"  (warna: success / hijau)
nilai >= 80  → grade = "B"  (warna: teal)
nilai >= 70  → grade = "C"  (warna: gold / emas)
nilai >= 60  → grade = "D"  (warna: warning / oranye)
nilai <  60  → grade = "E"  (warna: danger / merah)
```

### Tidak Ada Penalti

Soal yang dikosongkan tidak mengurangi poin. Hanya jawaban benar yang menambah poin.

---

## 10. Sistem Timer

Timer diimplementasikan menggunakan metode rekursif `_tick_timer()` yang memanfaatkan `tk.Frame.after()` — pendekatan non-blocking yang aman untuk GUI Tkinter.

```python
def _tick_timer(self):
    if self.selesai:
        return                       # Hentikan rekursi jika ujian selesai
    
    # Update tampilan
    self.lbl_timer.config(text=f"{menit:02d}:{detik:02d}")
    
    # Perubahan warna berdasarkan sisa waktu
    if   sisa_detik <= 300: warna = danger   # Merah
    elif sisa_detik <= 600: warna = warning  # Oranye
    else:                   warna = gold     # Emas
    
    if sisa_detik <= 0:
        # Waktu habis → dialog → _selesaikan()
        ...
    
    self.sisa_detik -= 1
    self.timer_job = self.after(1000, self._tick_timer)   # Rekursi 1 detik
```

Timer dapat dibatalkan menggunakan `self.after_cancel(self.timer_job)` saat ujian diselesaikan, mencegah kebocoran memori atau pemanggilan callback setelah layar dihancurkan.

---

## 11. Bank Soal — Ringkasan Konten

### Bagian A: Himpunan Bilangan (Soal 1–5)

| ID | Topik | Konsep Kunci |
|---|---|---|
| 1 | Klasifikasi Bilangan | Pemotongan π menghasilkan rasional |
| 2 | Bilangan Irasional | Pembuktian irasionalitas √5 |
| 3 | Hierarki Bilangan | Relasi ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ |
| 4 | Sifat Bilangan Real | Perkalian dua irasional tidak selalu irasional |
| 5 | Notasi Interval | Konversi pertidaksamaan ke notasi interval |

### Bagian B: Nilai Mutlak (Soal 6–10)

| ID | Topik | Konsep Kunci |
|---|---|---|
| 6 | Persamaan Nilai Mutlak | \|3x − 5\| = 7, dua kasus |
| 7 | Pertidaksamaan Nilai Mutlak | \|2x + 1\| < 5, satu interval |
| 8 | Pertidaksamaan Nilai Mutlak | \|x − 2\| > 3, dua interval |
| 9 | Sifat Nilai Mutlak | Ketidaksamaan segitiga |
| 10 | Aplikasi (Resistor) | Toleransi ±5% → model nilai mutlak |

### Bagian C: Pertidaksamaan (Soal 11–15)

| ID | Topik | Konsep Kunci |
|---|---|---|
| 11 | Linear | 5(x−2) ≤ 3(x+4) |
| 12 | Kuadrat | x² − 5x + 6 < 0, faktorisasi |
| 13 | Kuadrat | 2x² + x − 6 ≥ 0, rumus kuadrat |
| 14 | Rasional | (x−1)/(x+2) < 0, titik kritis |
| 15 | Aksioma Urutan | Pembalikan tanda saat dibagi negatif |

### Bagian D: Aplikasi & UTS (Soal 16–20)

| ID | Topik | Konsep Kunci |
|---|---|---|
| 16 | Op-Amp | Gain A=10, batas output → batas input |
| 17 | Transien RC | Vc(t) = 12(1−e^(−t/0,5)) > 6V |
| 18 | Aksioma Kelengkapan | Supremum vs maksimum |
| 19 | Operasi Interval | Irisan [1,5] ∩ [3,8] |
| 20 | Kombinasi | \|x+3\| ≤ 2x, dua kasus + syarat awal |

---

## 12. Diagram Alur Aplikasi

```
 ┌─────────────────────────────────────────────────────┐
 │                   AplikasiCAT.jalankan()            │
 └──────────────────────────┬──────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │        LayarLogin          │
              │  ─────────────────────     │
              │  [Nama]  [NIM]  [Kelas]   │
              │  [     MULAI UJIAN →    ]  │
              └─────────────┬─────────────┘
                            │ on_mulai(nama, nim, kelas)
              ┌─────────────▼─────────────┐
              │        LayarUjian          │
              │  ─────────────────────     │
              │  Header + Timer            │
              │  Sidebar Navigasi          │
              │  Area Soal                 │
              │   ├ Teks Pertanyaan        │
              │   ├ 4 Pilihan Radio        │
              │   └ Kontrol Nav/Tandai     │
              └─────────────┬─────────────┘
                            │ on_selesai(nama, nim, kelas,
                            │            jawaban, ditandai,
                            │            waktu_detik)
              ┌─────────────▼─────────────┐
              │        LayarRekap          │
              │  ─────────────────────     │
              │  Panel Nilai + Grade       │
              │  Statistik per Bagian      │
              │  Pembahasan per Soal       │
              │  [Ulangi]  [Keluar]        │
              └─────────────┬─────────────┘
                            │ on_ulang() → kembali ke LayarLogin
                            └──────────────────────────────────┘
```

---

## 13. Catatan Pengembangan

### Potensi Pengembangan Lanjutan

| Fitur | Kompleksitas | Keterangan |
|---|---|---|
| Pengacakan soal/pilihan | Rendah | Tambahkan `random.shuffle()` pada `BANK_SOAL` dan tiap `soal.pilihan` |
| Ekspor hasil ke PDF/CSV | Sedang | Integrasi `reportlab` atau `csv` module |
| Bank soal dari file JSON | Sedang | Pisahkan data soal ke file eksternal |
| Multi-bab | Sedang | Tambahkan tab atau menu pilihan bab |
| Penyimpanan riwayat | Tinggi | Integrasi SQLite atau file JSON |
| Pengamanan ujian | Tinggi | Enkripsi kunci, fullscreen lock, disable alt-tab |

### Catatan Implementasi

- **`indicatoron=False`** pada `Radiobutton`: merender tombol radio sebagai blok penuh (bukan radio kecil), memberikan target klik yang lebih besar dan estetika yang lebih modern.
- **`tk.Text` dengan `state="disabled"`**: digunakan untuk teks pertanyaan dan penjelasan agar tidak dapat diedit pengguna namun tetap dapat menampilkan teks panjang dengan *word-wrap*.
- **`canvas.bind_all("<MouseWheel>")`**: mengaktifkan scroll layar rekap menggunakan roda mouse di seluruh area, bukan hanya pada scrollbar.
- **Walrus operator** (`if soal_id := ...`): digunakan dalam `_update_nav()`, memerlukan Python 3.8+.
- **`pack_propagate(False)`**: digunakan pada sidebar untuk mempertahankan lebar tetap sebesar 200 piksel meskipun konten di dalamnya memiliki ukuran berbeda.

### Keterbatasan yang Diketahui

- Hasil tidak disimpan secara persisten setelah aplikasi ditutup.
- Bank soal tidak diacak antar sesi (urutan selalu sama).
- Tidak ada mekanisme anti-kecurangan.
- Performa scrollbar pada layar rekap dapat menurun pada resolusi atau DPI yang sangat tinggi.

---

*Dokumentasi ini mengacu pada `CATV2.py` — revisi terakhir yang diperiksa. Setiap perubahan kode berikutnya mungkin memerlukan pembaruan dokumentasi.*
