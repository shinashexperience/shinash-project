"""
╔══════════════════════════════════════════════════════════════════╗
║     CAT KALKULUS 1A — Bab 1: Sistem Bilangan Real               ║
║     S1 Teknik Elektro — Institut Teknologi Telkom Surabaya      ║
║     MAA1114 | GUI Edition | Tkinter                             ║
╚══════════════════════════════════════════════════════════════════╝

Cara Menjalankan:
    python3 CAT_Kalkulus1A_GUI.py

Tidak perlu install library tambahan — hanya menggunakan Tkinter
(sudah termasuk dalam instalasi Python standar).
"""

import tkinter as tk
from tkinter import ttk, messagebox, font
import time
import random
from dataclasses import dataclass, field
from typing import Optional
import math

# ═══════════════════════════════════════════════════════════════════
# DATA SOAL — 20 Soal Pilihan Ganda Kalkulus 1A Bab 1
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Soal:
    id: int
    bagian: str          # "A", "B", "C", "D"
    topik: str
    pertanyaan: str
    pilihan: list        # ["A. ...", "B. ...", ...]
    kunci: str           # "A", "B", "C", atau "D"
    penjelasan: str
    poin: int = 5

BANK_SOAL = [
    # ─── BAGIAN A: Himpunan Bilangan ───────────────────────────────
    Soal(
        id=1, bagian="A", topik="Klasifikasi Bilangan",
        pertanyaan="Bilangan 3,14159 (nilai π yang dipotong) termasuk ke dalam himpunan bilangan...",
        pilihan=[
            "A. Irasional, karena mengandung desimal",
            "B. Irasional, karena nilai π adalah irasional",
            "C. Rasional, karena desimalnya berhenti/terhingga",
            "D. Bulat, karena dapat ditulis tanpa pecahan"
        ],
        kunci="C",
        penjelasan=(
            "3,14159 adalah RASIONAL karena desimalnya berhenti (finite decimal).\n"
            "Bilangan rasional = p/q dengan p,q bilangan bulat dan q≠0.\n"
            "3,14159 = 314159/100000 → ini adalah rasional!\n\n"
            "Yang IRASIONAL adalah π sendiri (π = 3,14159265358979...∞ tak berulang).\n"
            "Ingat: pemotongan/pembulatan π menghasilkan bilangan RASIONAL, bukan irasional."
        ),
        poin=5
    ),
    Soal(
        id=2, bagian="A", topik="Bilangan Irasional",
        pertanyaan="Di antara pilihan berikut, manakah yang merupakan bilangan IRASIONAL?",
        pilihan=[
            "A. √4 = 2",
            "B. √9 = 3",
            "C. √5 ≈ 2,2360679...",
            "D. √16 = 4"
        ],
        kunci="C",
        penjelasan=(
            "√5 adalah IRASIONAL karena 5 bukan bilangan kuadrat sempurna.\n\n"
            "Pembuktian singkat: Jika √5 = p/q (tereduksi), maka:\n"
            "  5 = p²/q²  →  p² = 5q²  →  p kelipatan 5\n"
            "  p = 5k  →  q² = 5k²  →  q juga kelipatan 5\n"
            "  KONTRADIKSI! p dan q tidak bisa keduanya kelipatan 5.\n\n"
            "√4 = 2, √9 = 3, √16 = 4 → semuanya bilangan bulat (RASIONAL)."
        ),
        poin=5
    ),
    Soal(
        id=3, bagian="A", topik="Hierarki Bilangan",
        pertanyaan="Pernyataan manakah yang BENAR tentang hubungan himpunan bilangan?",
        pilihan=[
            "A. ℤ ⊂ ℕ (Bilangan bulat adalah bagian dari bilangan asli)",
            "B. ℝ ⊂ ℚ (Bilangan real adalah bagian dari bilangan rasional)",
            "C. ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ (Hierarki berurutan)",
            "D. ℚ dan ℝ adalah himpunan yang saling lepas"
        ],
        kunci="C",
        penjelasan=(
            "Hierarki BENAR adalah: ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ\n\n"
            "  ℕ = {1, 2, 3, ...} → bilangan asli\n"
            "  ℤ = {..., -2, -1, 0, 1, 2, ...} → mengandung ℕ plus negatif\n"
            "  ℚ = {p/q | p,q∈ℤ, q≠0} → mengandung ℤ plus pecahan\n"
            "  ℝ = ℚ ∪ I → mengandung ℚ plus irasional\n\n"
            "Pilihan A salah: ℕ ⊂ ℤ (bukan sebaliknya)\n"
            "Pilihan B salah: ℚ ⊂ ℝ (bukan sebaliknya)\n"
            "Pilihan D salah: ℚ ⊂ ℝ, jadi tidak saling lepas"
        ),
        poin=5
    ),
    Soal(
        id=4, bagian="A", topik="Sifat Bilangan Real",
        pertanyaan="Manakah pernyataan yang BENAR tentang √2 × √2?",
        pilihan=[
            "A. Irasional, karena perkalian dua irasional selalu irasional",
            "B. Rasional, karena hasilnya = 2 (bilangan bulat)",
            "C. Tidak terdefinisi dalam bilangan real",
            "D. Irasional, karena √2 adalah irasional"
        ],
        kunci="B",
        penjelasan=(
            "√2 × √2 = (√2)² = 2\n\n"
            "Hasilnya adalah 2, yaitu BILANGAN BULAT (RASIONAL)!\n\n"
            "Ini membuktikan bahwa:\n"
            "  'Perkalian dua bilangan irasional TIDAK SELALU irasional'\n\n"
            "Contoh lain:\n"
            "  √3 × √3 = 3 (rasional)\n"
            "  √2 × √8 = √16 = 4 (rasional)\n"
            "  Tetapi: √2 × √3 = √6 (irasional)\n\n"
            "Kunci: Lihat hasilnya, bukan sumber operandnya."
        ),
        poin=5
    ),
    Soal(
        id=5, bagian="A", topik="Notasi Interval",
        pertanyaan="Himpunan {x | -2 < x ≤ 5} dalam notasi interval adalah...",
        pilihan=[
            "A. [-2, 5]",
            "B. (-2, 5)",
            "C. (-2, 5]",
            "D. [-2, 5)"
        ],
        kunci="C",
        penjelasan=(
            "Analisis kondisi:\n"
            "  • -2 < x  →  x lebih BESAR dari -2, jadi -2 TIDAK termasuk → kurung TERBUKA (\n"
            "  • x ≤ 5   →  x KURANG DARI SAMA DENGAN 5, jadi 5 TERMASUK → kurung TERTUTUP ]\n\n"
            "Maka notasi intervalnya: (-2, 5]\n\n"
            "Aturan:\n"
            "  < atau > → kurung terbuka ( )\n"
            "  ≤ atau ≥ → kurung tertutup [ ]\n"
            "  ∞ → SELALU kurung terbuka (∞ bukan bilangan!)"
        ),
        poin=5
    ),

    # ─── BAGIAN B: Nilai Mutlak ─────────────────────────────────────
    Soal(
        id=6, bagian="B", topik="Persamaan Nilai Mutlak",
        pertanyaan="Penyelesaian dari |3x - 5| = 7 adalah...",
        pilihan=[
            "A. x = 4 saja",
            "B. x = -2/3 saja",
            "C. x = 4 atau x = -2/3",
            "D. x = 4 dan x = -2/3 (harus keduanya)"
        ],
        kunci="C",
        penjelasan=(
            "|3x - 5| = 7 berarti ekspresi dalam mutlak = 7 atau = -7\n\n"
            "Kasus 1: 3x - 5 = 7\n"
            "  3x = 12  →  x = 4 ✓\n\n"
            "Kasus 2: 3x - 5 = -7\n"
            "  3x = -2  →  x = -2/3 ✓\n\n"
            "Verifikasi:\n"
            "  |3(4) - 5| = |12-5| = |7| = 7 ✓\n"
            "  |3(-2/3) - 5| = |-2-5| = |-7| = 7 ✓\n\n"
            "Catatan: 'atau' (bukan 'dan') karena x memenuhi salah satu kasus."
        ),
        poin=5
    ),
    Soal(
        id=7, bagian="B", topik="Pertidaksamaan Nilai Mutlak",
        pertanyaan="Solusi dari |2x + 1| < 5 adalah...",
        pilihan=[
            "A. x > 2 atau x < -3",
            "B. -3 < x < 2",
            "C. x < -3 atau x > 2",
            "D. x < 2"
        ],
        kunci="B",
        penjelasan=(
            "|2x + 1| < 5\n\n"
            "Aturan: |ekspresi| < c  ⟺  -c < ekspresi < c\n\n"
            "Maka: -5 < 2x + 1 < 5\n\n"
            "Kurangi 1 semua sisi:\n"
            "  -6 < 2x < 4\n\n"
            "Bagi 2 semua sisi (positif, tanda tetap):\n"
            "  -3 < x < 2\n\n"
            "Solusi: (-3, 2)\n\n"
            "Ingat:\n"
            "  |x| < c → satu interval (AND)\n"
            "  |x| > c → dua interval (OR)"
        ),
        poin=5
    ),
    Soal(
        id=8, bagian="B", topik="Pertidaksamaan Nilai Mutlak",
        pertanyaan="Solusi dari |x - 2| > 3 dalam notasi interval adalah...",
        pilihan=[
            "A. (-1, 5)",
            "B. (-∞, -1) ∪ (5, +∞)",
            "C. (-∞, -1] ∪ [5, +∞)",
            "D. (-3, 3)"
        ],
        kunci="B",
        penjelasan=(
            "|x - 2| > 3\n\n"
            "Aturan: |ekspresi| > c  ⟺  ekspresi < -c atau ekspresi > c\n\n"
            "Kasus 1: x - 2 < -3  →  x < -1\n"
            "Kasus 2: x - 2 > 3   →  x > 5\n\n"
            "Karena > (bukan ≥), ujung tidak termasuk → kurung TERBUKA\n\n"
            "Solusi: (-∞, -1) ∪ (5, +∞)\n\n"
            "Pilihan C salah karena pakai kurung tertutup (seharusnya terbuka untuk >)\n"
            "Pilihan A salah karena itu adalah solusi |x-2| < 3"
        ),
        poin=5
    ),
    Soal(
        id=9, bagian="B", topik="Sifat Nilai Mutlak",
        pertanyaan="Manakah pernyataan yang BENAR tentang nilai mutlak?",
        pilihan=[
            "A. |a + b| = |a| + |b| selalu berlaku",
            "B. √(a²) = a selalu berlaku",
            "C. |a + b| ≤ |a| + |b| (Ketidaksamaan Segitiga)",
            "D. |a · b| = |a| - |b| selalu berlaku"
        ],
        kunci="C",
        penjelasan=(
            "Ketidaksamaan Segitiga: |a + b| ≤ |a| + |b|  ✓ SELALU BENAR\n\n"
            "Pilihan A SALAH: |a + b| = |a| + |b| hanya jika a,b sama tanda\n"
            "  Contoh: a=3, b=-3 → |3+(-3)| = 0 ≠ |3|+|-3| = 6\n\n"
            "Pilihan B SALAH: √(a²) = |a|, bukan a\n"
            "  Contoh: a=-3 → √((-3)²) = √9 = 3 = |-3|, bukan -3!\n\n"
            "Pilihan D SALAH: |a · b| = |a| · |b| (perkalian, bukan pengurangan)\n\n"
            "Pembuktian KS:\n"
            "  a ≤ |a| dan b ≤ |b|\n"
            "  → a+b ≤ |a|+|b|\n"
            "  Demikian juga -(a+b) ≤ |a|+|b|\n"
            "  → |a+b| ≤ |a|+|b| ✓"
        ),
        poin=5
    ),
    Soal(
        id=10, bagian="B", topik="Aplikasi Nilai Mutlak",
        pertanyaan=(
            "Sebuah resistor 1 kΩ memiliki toleransi ±5%. "
            "Rentang nilai resistansi yang masih diterima adalah..."
        ),
        pilihan=[
            "A. 950 Ω ≤ R ≤ 1050 Ω",
            "B. 900 Ω ≤ R ≤ 1100 Ω",
            "C. 995 Ω ≤ R ≤ 1005 Ω",
            "D. 500 Ω ≤ R ≤ 1500 Ω"
        ],
        kunci="A",
        penjelasan=(
            "Toleransi ±5% dari 1 kΩ = 1000 Ω:\n\n"
            "  Simpangan = 5% × 1000 = 50 Ω\n\n"
            "Dalam notasi nilai mutlak:\n"
            "  |R - 1000| ≤ 50\n\n"
            "Diselesaikan:\n"
            "  -50 ≤ R - 1000 ≤ 50\n"
            "  950 ≤ R ≤ 1050\n\n"
            "Jadi rentang: 950 Ω ≤ R ≤ 1050 Ω\n\n"
            "Aplikasi Nyata:\n"
            "  • Resistor dengan kode warna emas = toleransi 5%\n"
            "  • Resistor dengan kode warna perak = toleransi 10%\n"
            "  • Nilai mutlak langsung memodelkan toleransi komponen!"
        ),
        poin=5
    ),

    # ─── BAGIAN C: Pertidaksamaan ───────────────────────────────────
    Soal(
        id=11, bagian="C", topik="Pertidaksamaan Linear",
        pertanyaan="Solusi dari 5(x - 2) ≤ 3(x + 4) adalah...",
        pilihan=[
            "A. x ≤ 11",
            "B. x ≥ 11",
            "C. x ≤ -11",
            "D. x ≥ -11"
        ],
        kunci="A",
        penjelasan=(
            "5(x - 2) ≤ 3(x + 4)\n\n"
            "Ekspansi:\n"
            "  5x - 10 ≤ 3x + 12\n\n"
            "Pindahkan suku x ke kiri:\n"
            "  5x - 3x ≤ 12 + 10\n"
            "  2x ≤ 22\n\n"
            "Bagi 2 (positif, tanda tetap):\n"
            "  x ≤ 11\n\n"
            "Solusi: (-∞, 11]\n\n"
            "Verifikasi dengan x = 11:\n"
            "  Kiri: 5(11-2) = 5(9) = 45\n"
            "  Kanan: 3(11+4) = 3(15) = 45\n"
            "  45 ≤ 45 ✓"
        ),
        poin=5
    ),
    Soal(
        id=12, bagian="C", topik="Pertidaksamaan Kuadrat",
        pertanyaan="Solusi dari x² - 5x + 6 < 0 adalah...",
        pilihan=[
            "A. x < 2 atau x > 3",
            "B. (2, 3)",
            "C. (-∞, 2) ∪ (3, +∞)",
            "D. [2, 3]"
        ],
        kunci="B",
        penjelasan=(
            "x² - 5x + 6 < 0\n\n"
            "Langkah 1 — Faktorkan:\n"
            "  (x - 2)(x - 3) < 0\n\n"
            "Langkah 2 — Cari akar: x = 2 dan x = 3\n\n"
            "Langkah 3 — Analisis tanda:\n"
            "  • x < 2: misalnya x=0 → (0-2)(0-3) = (-2)(-3) = +6 > 0 ✗\n"
            "  • 2 < x < 3: misalnya x=2,5 → (0,5)(-0,5) = -0,25 < 0 ✓\n"
            "  • x > 3: misalnya x=4 → (2)(1) = +2 > 0 ✗\n\n"
            "Solusi: 2 < x < 3, yaitu interval (2, 3)\n\n"
            "Karena < (bukan ≤), titik ujung x=2 dan x=3 tidak termasuk."
        ),
        poin=5
    ),
    Soal(
        id=13, bagian="C", topik="Pertidaksamaan Kuadrat",
        pertanyaan="Solusi dari 2x² + x - 6 ≥ 0 adalah...",
        pilihan=[
            "A. [-2, 3/2]",
            "B. (-∞, -2) ∪ (3/2, +∞)",
            "C. (-∞, -2] ∪ [3/2, +∞)",
            "D. (-2, 3/2)"
        ],
        kunci="C",
        penjelasan=(
            "2x² + x - 6 ≥ 0\n\n"
            "Langkah 1 — Cari akar dengan rumus kuadrat:\n"
            "  x = (-1 ± √(1+48)) / 4 = (-1 ± 7) / 4\n"
            "  x₁ = (-1+7)/4 = 6/4 = 3/2\n"
            "  x₂ = (-1-7)/4 = -8/4 = -2\n\n"
            "Faktorisasi: 2(x - 3/2)(x + 2) ≥ 0 → (2x-3)(x+2) ≥ 0\n\n"
            "Langkah 2 — Analisis tanda:\n"
            "  • x < -2: positif ✓\n"
            "  • -2 < x < 3/2: negatif ✗\n"
            "  • x > 3/2: positif ✓\n\n"
            "Karena ≥ 0, ujung TERMASUK (kurung tertutup)\n\n"
            "Solusi: (-∞, -2] ∪ [3/2, +∞)"
        ),
        poin=5
    ),
    Soal(
        id=14, bagian="C", topik="Pertidaksamaan Rasional",
        pertanyaan="Solusi dari (x - 1)/(x + 2) < 0 adalah...",
        pilihan=[
            "A. (-∞, -2) ∪ (1, +∞)",
            "B. [-2, 1]",
            "C. (-2, 1)",
            "D. (-∞, -2] ∪ [1, +∞)"
        ],
        kunci="C",
        penjelasan=(
            "(x - 1)/(x + 2) < 0\n\n"
            "Titik kritis: x = 1 (pembilang = 0) dan x = -2 (penyebut = 0)\n\n"
            "Analisis tanda di 3 interval:\n"
            "  • x < -2: contoh x=-3 → (-3-1)/(-3+2) = (-4)/(-1) = 4 > 0 ✗\n"
            "  • -2 < x < 1: contoh x=0 → (0-1)/(0+2) = (-1)/(2) = -0,5 < 0 ✓\n"
            "  • x > 1: contoh x=2 → (2-1)/(2+2) = (1)/(4) = 0,25 > 0 ✗\n\n"
            "Hanya interval (-2, 1) yang memenuhi.\n\n"
            "PERHATIAN:\n"
            "  • x = -2: TIDAK termasuk (penyebut = 0, tidak terdefinisi!)\n"
            "  • x = 1: TIDAK termasuk (persamaannya < 0, bukan ≤ 0)\n\n"
            "Solusi: (-2, 1)"
        ),
        poin=5
    ),
    Soal(
        id=15, bagian="C", topik="Aksioma Urutan",
        pertanyaan=(
            "Saat menyelesaikan pertidaksamaan -3x + 6 > 0, "
            "langkah membagi -3 ke semua sisi menghasilkan..."
        ),
        pilihan=[
            "A. x > -2 (tanda tetap karena -3 adalah konstanta)",
            "B. x < 2 (tanda BERBALIK karena dibagi negatif)",
            "C. x > 2 (tanda tetap)",
            "D. x < -2 (tanda tetap)"
        ],
        kunci="B",
        penjelasan=(
            "-3x + 6 > 0\n\n"
            "Kurangi 6 kedua sisi:\n"
            "  -3x > -6\n\n"
            "Bagi -3 kedua sisi — PERHATIAN: bagi NEGATIF, tanda BERBALIK!\n"
            "  x < 2  (< bukan >, karena dibagi negatif)\n\n"
            "Ini adalah AKSIOMA URUTAN:\n"
            "  Jika a < b dan c < 0, maka ac > bc\n\n"
            "Verifikasi:\n"
            "  -3(1) + 6 = 3 > 0 ✓ (x=1 < 2, harusnya memenuhi)\n"
            "  -3(3) + 6 = -3 > 0 ✗ (x=3 > 2, harusnya tidak memenuhi)\n\n"
            "Ini adalah KESALAHAN PALING SERING dalam pertidaksamaan!"
        ),
        poin=5
    ),

    # ─── BAGIAN D: Aplikasi & UTS ──────────────────────────────────
    Soal(
        id=16, bagian="D", topik="Aplikasi Op-Amp",
        pertanyaan=(
            "Op-Amp dengan gain A = 10 memiliki keterbatasan output |Vout| < 13 V. "
            "Rentang tegangan input Vin yang AMAN (tidak mengalami clipping) adalah..."
        ),
        pilihan=[
            "A. |Vin| < 130 V",
            "B. |Vin| < 1,3 V",
            "C. |Vin| < 13 V",
            "D. |Vin| < 0,13 V"
        ],
        kunci="B",
        penjelasan=(
            "Diketahui:\n"
            "  • Gain A = 10\n"
            "  • Batas output: |Vout| < 13 V\n"
            "  • Hubungan: Vout = A × Vin = 10 × Vin\n\n"
            "Substitusi:\n"
            "  |10 × Vin| < 13\n"
            "  10 × |Vin| < 13     (sifat nilai mutlak: |ab| = |a||b|)\n"
            "  |Vin| < 13/10\n"
            "  |Vin| < 1,3 V\n\n"
            "Artinya: -1,3 V < Vin < 1,3 V\n\n"
            "Jika sinyal input = 2 sin(2πt), amplitudo = 2V > 1,3V\n"
            "→ AKAN TERJADI CLIPPING (distorsi sinyal)!\n\n"
            "Aplikasi: nilai mutlak memodelkan batas simetris tegangan."
        ),
        poin=5
    ),
    Soal(
        id=17, bagian="D", topik="Transien RC",
        pertanyaan=(
            "Rangkaian RC dengan τ = 0,5 s dan Vs = 12 V. "
            "Kapasitor mengisi dengan Vc(t) = 12(1 - e^(-t/0,5)). "
            "Pada waktu berapakah Vc > 6 V?"
        ),
        pilihan=[
            "A. t > 0,693 s",
            "B. t > 0,5 s",
            "C. t > 0,347 s",
            "D. t > 1,0 s"
        ],
        kunci="C",
        penjelasan=(
            "Vc(t) = 12(1 - e^(-t/0,5)) > 6\n\n"
            "Bagi 12:\n"
            "  1 - e^(-2t) > 0,5\n\n"
            "Kurangi 1:\n"
            "  -e^(-2t) > -0,5\n\n"
            "Kali -1 (TANDA BERBALIK!):\n"
            "  e^(-2t) < 0,5\n\n"
            "Ambil ln kedua sisi:\n"
            "  -2t < ln(0,5) = -0,693\n\n"
            "Bagi -2 (TANDA BERBALIK!):\n"
            "  t > 0,3465 ≈ 0,347 s\n\n"
            "Jawaban: t > 0,347 s\n\n"
            "Kunci: Ada DUA kali pembalikan tanda:\n"
            "  1. Saat kali -1 untuk e^(-2t)\n"
            "  2. Saat bagi -2 di akhir"
        ),
        poin=5
    ),
    Soal(
        id=18, bagian="D", topik="Aksioma Kelengkapan",
        pertanyaan=(
            "Aksioma Kelengkapan (Completeness Axiom) bilangan real menyatakan bahwa "
            "setiap himpunan bagian ℝ yang tidak kosong dan terbatas atas memiliki..."
        ),
        pilihan=[
            "A. Nilai maksimum (maximum)",
            "B. Supremum (batas atas terkecil)",
            "C. Infimum (batas bawah terbesar)",
            "D. Rata-rata (average)"
        ],
        kunci="B",
        penjelasan=(
            "Aksioma Kelengkapan: Setiap himpunan S ⊆ ℝ yang:\n"
            "  (1) Tidak kosong\n"
            "  (2) Terbatas atas (ada M dengan s ≤ M untuk semua s ∈ S)\n"
            "...memiliki SUPREMUM (batas atas terkecil).\n\n"
            "Supremum ≠ Nilai Maksimum:\n"
            "  • Maksimum: elemen terbesar yang ada di dalam S\n"
            "  • Supremum: batas atas terkecil (boleh tidak ada di S)\n\n"
            "Contoh:\n"
            "  S = (0, 1) → sup S = 1, tapi max S tidak ada!\n\n"
            "Mengapa penting?\n"
            "  → ℚ TIDAK memenuhi ini! (Lubang di bilangan rasional)\n"
            "  → Ini yang membuat ℝ 'lengkap' dan fondasi LIMIT."
        ),
        poin=5
    ),
    Soal(
        id=19, bagian="D", topik="Operasi Interval",
        pertanyaan="Hasil dari [1, 5] ∩ [3, 8] adalah...",
        pilihan=[
            "A. [1, 8]",
            "B. [3, 5]",
            "C. [1, 3] ∪ [5, 8]",
            "D. ∅ (himpunan kosong)"
        ],
        kunci="B",
        penjelasan=(
            "Irisan (∩) = ambil bagian yang ada di KEDUA interval.\n\n"
            "  [1, 5] = semua x dengan 1 ≤ x ≤ 5\n"
            "  [3, 8] = semua x dengan 3 ≤ x ≤ 8\n\n"
            "Yang ada di KEDUANYA: 3 ≤ x ≤ 5\n\n"
            "Jadi [1, 5] ∩ [3, 8] = [3, 5]\n\n"
            "Visualisasi:\n"
            "  [──────────]\n"
            "  1    3  5\n"
            "       [───────────]\n"
            "       3    5    8\n"
            "       [────]\n"
            "       3  5  ← Irisan\n\n"
            "Pilihan A adalah GABUNGAN [1,5] ∪ [3,8] = [1,8]"
        ),
        poin=5
    ),
    Soal(
        id=20, bagian="D", topik="Pertidaksamaan + Nilai Mutlak",
        pertanyaan="Nilai x yang memenuhi |x + 3| ≤ 2x adalah...",
        pilihan=[
            "A. x ≥ 1",
            "B. x ≥ 3",
            "C. 1 ≤ x ≤ 3",
            "D. x ≤ 1 atau x ≥ 3"
        ],
        kunci="B",
        penjelasan=(
            "|x + 3| ≤ 2x\n\n"
            "Syarat awal: 2x ≥ 0 → x ≥ 0 (nilai mutlak ≥ 0)\n\n"
            "Pecah menjadi 2 kasus:\n\n"
            "Kasus 1 (x + 3 ≥ 0, yaitu x ≥ -3):\n"
            "  x + 3 ≤ 2x  →  3 ≤ x  →  x ≥ 3\n\n"
            "Kasus 2 (x + 3 < 0, yaitu x < -3):\n"
            "  -(x + 3) ≤ 2x  →  -x - 3 ≤ 2x  →  -3 ≤ 3x  →  x ≥ -1\n"
            "  Tapi di kasus ini x < -3, tidak ada irisan. ✗\n\n"
            "Gabungkan dengan syarat x ≥ 0:\n"
            "  Dari kasus 1: x ≥ 3 (sudah mencakup x ≥ 0)\n\n"
            "Solusi: x ≥ 3, yaitu [3, +∞)\n\n"
            "Verifikasi:\n"
            "  x=3: |3+3|=6 ≤ 2(3)=6 ✓\n"
            "  x=5: |5+3|=8 ≤ 2(5)=10 ✓\n"
            "  x=1: |1+3|=4 ≤ 2(1)=2 ✗ (tidak memenuhi)"
        ),
        poin=5
    ),
]

# ═══════════════════════════════════════════════════════════════════
# KONFIGURASI UJIAN
# ═══════════════════════════════════════════════════════════════════

DURASI_UJIAN_DETIK = 40 * 60   # 40 menit
NAMA_UJIAN = "Kalkulus 1A — Bab 1: Sistem Bilangan Real"
KODE_MK = "MAA1114"
PRODI = "S1 Teknik Elektro"
INSTITUSI = "Institut Teknologi Telkom Surabaya"

# ═══════════════════════════════════════════════════════════════════
# PALET WARNA
# ═══════════════════════════════════════════════════════════════════

CLR = {
    # Latar & Panel
    "bg_dark":     "#0F1117",
    "bg_panel":    "#1A1D2E",
    "bg_card":     "#1E2235",
    "bg_input":    "#252840",
    "bg_hover":    "#2D3154",

    # Aksen
    "accent":      "#4C6EF5",
    "accent_glow": "#5C7CFA",
    "success":     "#40C057",
    "warning":     "#FD7E14",
    "danger":      "#FA5252",
    "gold":        "#FFD43B",
    "teal":        "#20C997",
    "purple":      "#9775FA",

    # Teks
    "text_bright": "#FFFFFF",
    "text_main":   "#CDD3F0",
    "text_dim":    "#7A84B3",
    "text_muted":  "#4A5280",

    # Border
    "border":      "#2D3154",
    "border_glow": "#4C6EF5",
}

# ═══════════════════════════════════════════════════════════════════
# LAYAR: LOGIN / IDENTITAS PESERTA
# ═══════════════════════════════════════════════════════════════════

class LayarLogin(tk.Frame):
    def __init__(self, master, on_mulai):
        super().__init__(master, bg=CLR["bg_dark"])
        self.on_mulai = on_mulai
        self._bangun()

    def _bangun(self):
        self.pack(fill="both", expand=True)

        # Canvas background dengan pattern
        canvas = tk.Canvas(self, bg=CLR["bg_dark"], highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Draw decorative dots
        for i in range(0, 800, 40):
            for j in range(0, 700, 40):
                canvas.create_oval(i-1, j-1, i+1, j+1,
                                   fill=CLR["text_muted"], outline="")

        # Center frame
        frame = tk.Frame(canvas, bg=CLR["bg_panel"],
                         relief="flat", bd=0)
        frame.place(relx=0.5, rely=0.5, anchor="center", width=480, height=520)

        # Top accent bar
        tk.Frame(frame, bg=CLR["accent"], height=4).pack(fill="x")

        # Logo area
        logo_frame = tk.Frame(frame, bg=CLR["bg_panel"], pady=24)
        logo_frame.pack(fill="x")

        tk.Label(logo_frame, text="🎓", font=("Segoe UI Emoji", 36),
                 bg=CLR["bg_panel"], fg=CLR["accent"]).pack()
        tk.Label(logo_frame, text="COMPUTER ASSISTED TEST",
                 font=("Consolas", 11, "bold"), bg=CLR["bg_panel"],
                 fg=CLR["accent"]).pack(pady=(4, 0))
        tk.Label(logo_frame, text=NAMA_UJIAN,
                 font=("Segoe UI", 13, "bold"), bg=CLR["bg_panel"],
                 fg=CLR["text_bright"], wraplength=400).pack(pady=(4, 0))
        tk.Label(logo_frame, text=f"{KODE_MK}  •  {PRODI}",
                 font=("Consolas", 9), bg=CLR["bg_panel"],
                 fg=CLR["text_dim"]).pack(pady=(2, 0))

        # Divider
        tk.Frame(frame, bg=CLR["border"], height=1).pack(fill="x", padx=24)

        # Form area
        form = tk.Frame(frame, bg=CLR["bg_panel"], padx=36, pady=20)
        form.pack(fill="x")

        # Nama
        tk.Label(form, text="NAMA LENGKAP", font=("Consolas", 8, "bold"),
                 bg=CLR["bg_panel"], fg=CLR["text_dim"],
                 anchor="w").pack(fill="x", pady=(0, 4))
        self.entry_nama = tk.Entry(form, font=("Segoe UI", 11),
                                   bg=CLR["bg_input"], fg=CLR["text_bright"],
                                   insertbackground=CLR["accent"],
                                   relief="flat", bd=0)
        self.entry_nama.pack(fill="x", ipady=10)
        tk.Frame(form, bg=CLR["accent"], height=2).pack(fill="x")

        # NIM
        tk.Label(form, text="NIM", font=("Consolas", 8, "bold"),
                 bg=CLR["bg_panel"], fg=CLR["text_dim"],
                 anchor="w").pack(fill="x", pady=(16, 4))
        self.entry_nim = tk.Entry(form, font=("Segoe UI", 11),
                                  bg=CLR["bg_input"], fg=CLR["text_bright"],
                                  insertbackground=CLR["accent"],
                                  relief="flat", bd=0)
        self.entry_nim.pack(fill="x", ipady=10)
        tk.Frame(form, bg=CLR["accent"], height=2).pack(fill="x")

        # Kelas
        tk.Label(form, text="KELAS", font=("Consolas", 8, "bold"),
                 bg=CLR["bg_panel"], fg=CLR["text_dim"],
                 anchor="w").pack(fill="x", pady=(16, 4))
        self.combo_kelas = ttk.Combobox(form, font=("Segoe UI", 11),
                                        values=["TE-A", "TE-B", "TE-C",
                                                "TE-D", "TE-E", "Lainnya"],
                                        state="readonly")
        self.combo_kelas.pack(fill="x", ipady=5)
        self.combo_kelas.current(0)

        # Info ujian
        info_frame = tk.Frame(form, bg=CLR["bg_card"], padx=12, pady=8)
        info_frame.pack(fill="x", pady=(16, 0))
        infos = [
            ("📝 Jumlah Soal", "20 Soal Pilihan Ganda"),
            ("⏱ Durasi",       "40 Menit"),
            ("🏆 Total Poin",   "100 Poin"),
        ]
        for label, val in infos:
            row = tk.Frame(info_frame, bg=CLR["bg_card"])
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, font=("Segoe UI", 9),
                     bg=CLR["bg_card"], fg=CLR["text_dim"],
                     width=18, anchor="w").pack(side="left")
            tk.Label(row, text=val, font=("Segoe UI", 9, "bold"),
                     bg=CLR["bg_card"], fg=CLR["text_main"],
                     anchor="w").pack(side="left")

        # Tombol mulai
        btn = tk.Button(form, text="MULAI UJIAN  →",
                        font=("Consolas", 11, "bold"),
                        bg=CLR["accent"], fg=CLR["text_bright"],
                        activebackground=CLR["accent_glow"],
                        activeforeground=CLR["text_bright"],
                        cursor="hand2", relief="flat", bd=0,
                        command=self._validasi_dan_mulai)
        btn.pack(fill="x", pady=(20, 4), ipady=12)

        self.entry_nama.focus()
        self.entry_nama.bind("<Return>", lambda e: self.entry_nim.focus())
        self.entry_nim.bind("<Return>",  lambda e: self._validasi_dan_mulai())

    def _validasi_dan_mulai(self):
        nama = self.entry_nama.get().strip()
        nim  = self.entry_nim.get().strip()
        kelas = self.combo_kelas.get()
        if not nama:
            messagebox.showwarning("Peringatan", "Nama lengkap wajib diisi!")
            return
        if not nim:
            messagebox.showwarning("Peringatan", "NIM wajib diisi!")
            return
        self.on_mulai(nama, nim, kelas)


# ═══════════════════════════════════════════════════════════════════
# LAYAR: UJIAN UTAMA
# ═══════════════════════════════════════════════════════════════════

class LayarUjian(tk.Frame):
    def __init__(self, master, nama, nim, kelas, on_selesai):
        super().__init__(master, bg=CLR["bg_dark"])
        self.nama = nama
        self.nim  = nim
        self.kelas = kelas
        self.on_selesai = on_selesai

        self.soal_list = BANK_SOAL[:]
        self.total = len(self.soal_list)
        self.idx_aktif = 0
        self.jawaban = {}       # {soal_id: pilihan "A"/"B"/"C"/"D"}
        self.ditandai = set()   # soal_id yang di-flag/tandai
        self.sisa_detik = DURASI_UJIAN_DETIK
        self.selesai = False
        self.timer_job = None

        self._bangun()
        self._tampilkan_soal(0)
        self._tick_timer()

    # ─── Layout ───────────────────────────────────────────────────
    def _bangun(self):
        self.pack(fill="both", expand=True)

        # ── Header Bar ──
        hdr = tk.Frame(self, bg=CLR["bg_panel"], height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text=f"  🎓 CAT  •  {NAMA_UJIAN}",
                 font=("Consolas", 9, "bold"),
                 bg=CLR["bg_panel"], fg=CLR["text_dim"],
                 anchor="w").pack(side="left", padx=8)

        timer_frame = tk.Frame(hdr, bg=CLR["bg_card"], padx=14, pady=8)
        timer_frame.pack(side="right", padx=12, pady=8)
        tk.Label(timer_frame, text="⏱", font=("Segoe UI Emoji", 11),
                 bg=CLR["bg_card"], fg=CLR["text_dim"]).pack(side="left")
        self.lbl_timer = tk.Label(timer_frame, text="40:00",
                                   font=("Consolas", 15, "bold"),
                                   bg=CLR["bg_card"], fg=CLR["gold"])
        self.lbl_timer.pack(side="left", padx=(4, 0))

        tk.Label(hdr, text=f"  {self.nama}  |  {self.nim}",
                 font=("Segoe UI", 9), bg=CLR["bg_panel"],
                 fg=CLR["text_dim"]).pack(side="right", padx=(0, 8))

        tk.Frame(self, bg=CLR["border"], height=1).pack(fill="x")

        # ── Body ──
        body = tk.Frame(self, bg=CLR["bg_dark"])
        body.pack(fill="both", expand=True)

        # Sidebar kiri: navigasi soal
        self.sidebar = tk.Frame(body, bg=CLR["bg_panel"], width=200)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="NAVIGASI SOAL",
                 font=("Consolas", 8, "bold"),
                 bg=CLR["bg_panel"], fg=CLR["text_muted"],
                 anchor="w").pack(fill="x", padx=12, pady=(14, 6))

        # Grid tombol soal
        nav_scroll = tk.Frame(self.sidebar, bg=CLR["bg_panel"])
        nav_scroll.pack(fill="both", expand=True, padx=8)

        self.btn_nav = []
        bagian_sekarang = None
        for i, soal in enumerate(self.soal_list):
            if soal.bagian != bagian_sekarang:
                bagian_sekarang = soal.bagian
                lbl = tk.Label(nav_scroll,
                               text=f"  Bagian {soal.bagian}",
                               font=("Consolas", 8, "bold"),
                               bg=CLR["bg_panel"],
                               fg=CLR["accent"],
                               anchor="w")
                lbl.pack(fill="x", pady=(8, 2))

            btn = tk.Button(
                nav_scroll,
                text=str(i + 1),
                font=("Consolas", 9, "bold"),
                width=3, height=1,
                bg=CLR["bg_input"], fg=CLR["text_dim"],
                activebackground=CLR["accent"],
                activeforeground="white",
                relief="flat", bd=0, cursor="hand2",
                command=lambda idx=i: self._tampilkan_soal(idx)
            )
            btn.pack(side="left", padx=2, pady=2)
            self.btn_nav.append(btn)

        # Legenda
        legend = tk.Frame(self.sidebar, bg=CLR["bg_panel"], pady=10)
        legend.pack(fill="x", padx=10)
        legends = [
            (CLR["accent"],   "Aktif"),
            (CLR["success"],  "Dijawab"),
            (CLR["warning"],  "Ditandai"),
            (CLR["bg_input"], "Belum"),
        ]
        for color, text in legends:
            row = tk.Frame(legend, bg=CLR["bg_panel"])
            row.pack(fill="x", pady=1)
            tk.Frame(row, bg=color, width=12, height=12).pack(side="left")
            tk.Label(row, text=f"  {text}", font=("Segoe UI", 8),
                     bg=CLR["bg_panel"], fg=CLR["text_dim"]).pack(side="left")

        # Progress bar mini
        self.lbl_progress = tk.Label(self.sidebar,
                                      text="0 / 20 dijawab",
                                      font=("Consolas", 8),
                                      bg=CLR["bg_panel"],
                                      fg=CLR["text_dim"])
        self.lbl_progress.pack(pady=(0, 4))

        tk.Frame(self.sidebar, bg=CLR["border"], height=1).pack(fill="x", padx=8)

        btn_selesai = tk.Button(
            self.sidebar, text="SELESAIKAN UJIAN",
            font=("Consolas", 8, "bold"),
            bg=CLR["danger"], fg="white",
            activebackground="#E03131",
            relief="flat", bd=0, cursor="hand2",
            command=self._konfirmasi_selesai
        )
        btn_selesai.pack(fill="x", padx=10, pady=10, ipady=8)

        # Area soal utama
        self.area_soal = tk.Frame(body, bg=CLR["bg_dark"])
        self.area_soal.pack(side="left", fill="both", expand=True, padx=20, pady=16)

        # Card soal
        self.card = tk.Frame(self.area_soal, bg=CLR["bg_card"],
                              relief="flat", bd=0)
        self.card.pack(fill="both", expand=True)

        # Top strip warna bagian
        self.strip_bagian = tk.Frame(self.card, height=4)
        self.strip_bagian.pack(fill="x")

        soal_body = tk.Frame(self.card, bg=CLR["bg_card"])
        soal_body.pack(fill="both", expand=True, padx=28, pady=20)

        # Header soal
        hdr_soal = tk.Frame(soal_body, bg=CLR["bg_card"])
        hdr_soal.pack(fill="x", pady=(0, 12))
        self.lbl_nomor = tk.Label(hdr_soal, text="Soal 1",
                                   font=("Consolas", 10, "bold"),
                                   bg=CLR["bg_card"], fg=CLR["accent"])
        self.lbl_nomor.pack(side="left")
        self.lbl_topik = tk.Label(hdr_soal, text="",
                                   font=("Segoe UI", 9),
                                   bg=CLR["bg_card"], fg=CLR["text_dim"])
        self.lbl_topik.pack(side="left", padx=(8, 0))
        self.lbl_poin = tk.Label(hdr_soal, text="5 poin",
                                  font=("Consolas", 9),
                                  bg=CLR["bg_card"], fg=CLR["gold"])
        self.lbl_poin.pack(side="right")

        # Teks pertanyaan
        self.txt_soal = tk.Text(soal_body,
                                 font=("Segoe UI", 12),
                                 bg=CLR["bg_card"], fg=CLR["text_bright"],
                                 relief="flat", bd=0,
                                 wrap="word", height=4,
                                 state="disabled",
                                 cursor="arrow")
        self.txt_soal.pack(fill="x", pady=(0, 16))

        # Pilihan jawaban
        self.var_jawaban = tk.StringVar(value="")
        self.radio_btns = []
        self.radio_frames = []

        for huruf in ["A", "B", "C", "D"]:
            f = tk.Frame(soal_body, bg=CLR["bg_input"],
                         cursor="hand2", pady=1)
            f.pack(fill="x", pady=4)

            rb = tk.Radiobutton(
                f,
                variable=self.var_jawaban,
                value=huruf,
                font=("Segoe UI", 11),
                bg=CLR["bg_input"], fg=CLR["text_main"],
                activebackground=CLR["bg_hover"],
                activeforeground=CLR["text_bright"],
                selectcolor=CLR["accent"],
                relief="flat", bd=0,
                anchor="w",
                indicatoron=False,
                command=self._simpan_jawaban,
                padx=16, pady=12
            )
            rb.pack(fill="x")
            self.radio_btns.append(rb)
            self.radio_frames.append(f)

        # Bottom controls
        ctrl = tk.Frame(soal_body, bg=CLR["bg_card"])
        ctrl.pack(fill="x", pady=(16, 0))

        self.btn_tandai = tk.Button(
            ctrl, text="🚩 Tandai Soal",
            font=("Segoe UI", 9),
            bg=CLR["bg_input"], fg=CLR["text_dim"],
            activebackground=CLR["warning"],
            activeforeground="white",
            relief="flat", bd=0, cursor="hand2",
            command=self._toggle_tandai,
            padx=12, pady=8
        )
        self.btn_tandai.pack(side="left")

        tk.Frame(ctrl, bg=CLR["bg_card"]).pack(side="left", expand=True)

        self.btn_prev = tk.Button(
            ctrl, text="← Sebelumnya",
            font=("Consolas", 9, "bold"),
            bg=CLR["bg_input"], fg=CLR["text_main"],
            activebackground=CLR["bg_hover"],
            activeforeground=CLR["text_bright"],
            relief="flat", bd=0, cursor="hand2",
            command=self._soal_prev,
            padx=16, pady=8
        )
        self.btn_prev.pack(side="left", padx=(0, 8))

        self.btn_next = tk.Button(
            ctrl, text="Berikutnya →",
            font=("Consolas", 9, "bold"),
            bg=CLR["accent"], fg="white",
            activebackground=CLR["accent_glow"],
            activeforeground="white",
            relief="flat", bd=0, cursor="hand2",
            command=self._soal_next,
            padx=16, pady=8
        )
        self.btn_next.pack(side="left")

    # ─── Tampilkan Soal ───────────────────────────────────────────
    WARNA_BAGIAN = {
        "A": CLR["accent"],
        "B": CLR["teal"],
        "C": CLR["purple"],
        "D": CLR["gold"],
    }

    def _tampilkan_soal(self, idx):
        self.idx_aktif = idx
        soal = self.soal_list[idx]

        warna = self.WARNA_BAGIAN.get(soal.bagian, CLR["accent"])
        self.strip_bagian.config(bg=warna)

        self.lbl_nomor.config(
            text=f"Soal {idx + 1} dari {self.total}  (Bagian {soal.bagian})",
            fg=warna
        )
        self.lbl_topik.config(text=f"— {soal.topik}")
        self.lbl_poin.config(text=f"{soal.poin} poin")

        self.txt_soal.config(state="normal")
        self.txt_soal.delete("1.0", "end")
        self.txt_soal.insert("end", soal.pertanyaan)
        self.txt_soal.config(state="disabled",
                              height=max(3, len(soal.pertanyaan) // 60 + 2))

        self.var_jawaban.set(self.jawaban.get(soal.id, ""))

        huruf_list = ["A", "B", "C", "D"]
        for i, (huruf, rb, f) in enumerate(zip(huruf_list, self.radio_btns, self.radio_frames)):
            teks = soal.pilihan[i]
            rb.config(text=teks, value=huruf)

            if self.var_jawaban.get() == huruf:
                f.config(bg=CLR["bg_hover"])
                rb.config(bg=CLR["bg_hover"], fg=CLR["text_bright"])
            else:
                f.config(bg=CLR["bg_input"])
                rb.config(bg=CLR["bg_input"], fg=CLR["text_main"])

        # Tombol tandai
        if soal.id in self.ditandai:
            self.btn_tandai.config(bg=CLR["warning"], fg="white",
                                    text="🚩 Ditandai")
        else:
            self.btn_tandai.config(bg=CLR["bg_input"], fg=CLR["text_dim"],
                                    text="🚩 Tandai Soal")

        # Navigasi prev/next
        self.btn_prev.config(state="normal" if idx > 0 else "disabled")
        if idx >= self.total - 1:
            self.btn_next.config(text="Selesaikan ✓",
                                  bg=CLR["success"], command=self._konfirmasi_selesai)
        else:
            self.btn_next.config(text="Berikutnya →",
                                  bg=CLR["accent"], command=self._soal_next)

        self._update_nav()

    def _simpan_jawaban(self):
        soal = self.soal_list[self.idx_aktif]
        jawab = self.var_jawaban.get()
        self.jawaban[soal.id] = jawab

        huruf_list = ["A", "B", "C", "D"]
        for huruf, rb, f in zip(huruf_list, self.radio_btns, self.radio_frames):
            if jawab == huruf:
                f.config(bg=CLR["bg_hover"])
                rb.config(bg=CLR["bg_hover"], fg=CLR["text_bright"])
            else:
                f.config(bg=CLR["bg_input"])
                rb.config(bg=CLR["bg_input"], fg=CLR["text_main"])

        self._update_nav()

    def _toggle_tandai(self):
        soal = self.soal_list[self.idx_aktif]
        if soal.id in self.ditandai:
            self.ditandai.remove(soal.id)
            self.btn_tandai.config(bg=CLR["bg_input"], fg=CLR["text_dim"],
                                    text="🚩 Tandai Soal")
        else:
            self.ditandai.add(soal.id)
            self.btn_tandai.config(bg=CLR["warning"], fg="white",
                                    text="🚩 Ditandai")
        self._update_nav()

    def _soal_prev(self):
        if self.idx_aktif > 0:
            self._tampilkan_soal(self.idx_aktif - 1)

    def _soal_next(self):
        if self.idx_aktif < self.total - 1:
            self._tampilkan_soal(self.idx_aktif + 1)

    # ─── Update Tombol Navigasi ────────────────────────────────────
    def _update_nav(self):
        dijawab = 0
        for i, soal in enumerate(self.soal_list):
            btn = self.btn_nav[i]
            sid = soal.id
            is_aktif   = (i == self.idx_aktif)
            is_dijawab = sid in self.jawaban
            is_tandai  = sid in self.ditandai

            if is_aktif:
                warna = self.WARNA_BAGIAN.get(soal.bagian, CLR["accent"])
                btn.config(bg=warna, fg="white")
            elif is_tandai:
                btn.config(bg=CLR["warning"], fg="white")
            elif is_dijawab:
                btn.config(bg=CLR["success"], fg="white")
                dijawab += 1
            else:
                btn.config(bg=CLR["bg_input"], fg=CLR["text_dim"])

        if soal_id := self.soal_list[self.idx_aktif].id:
            if soal_id in self.jawaban:
                dijawab = sum(1 for s in self.soal_list if s.id in self.jawaban)

        self.lbl_progress.config(
            text=f"{dijawab} / {self.total} dijawab"
        )

    # ─── Timer ────────────────────────────────────────────────────
    def _tick_timer(self):
        if self.selesai:
            return
        menit = self.sisa_detik // 60
        detik = self.sisa_detik % 60
        self.lbl_timer.config(text=f"{menit:02d}:{detik:02d}")

        if self.sisa_detik <= 300:        # 5 menit terakhir
            self.lbl_timer.config(fg=CLR["danger"])
        elif self.sisa_detik <= 600:      # 10 menit terakhir
            self.lbl_timer.config(fg=CLR["warning"])

        if self.sisa_detik <= 0:
            self.lbl_timer.config(text="00:00", fg=CLR["danger"])
            messagebox.showwarning(
                "Waktu Habis!",
                "Waktu ujian telah habis. Jawaban Anda akan dikumpulkan secara otomatis."
            )
            self._selesaikan()
            return

        self.sisa_detik -= 1
        self.timer_job = self.after(1000, self._tick_timer)

    # ─── Konfirmasi & Selesaikan ──────────────────────────────────
    def _konfirmasi_selesai(self):
        dijawab = sum(1 for s in self.soal_list if s.id in self.jawaban)
        belum   = self.total - dijawab
        tandai  = len(self.ditandai)

        pesan = (
            f"Anda akan mengumpulkan ujian.\n\n"
            f"  Dijawab  : {dijawab} soal\n"
            f"  Belum    : {belum} soal\n"
            f"  Ditandai : {tandai} soal\n\n"
            f"Lanjutkan?"
        )
        if messagebox.askyesno("Konfirmasi Selesai", pesan, default="no"):
            self._selesaikan()

    def _selesaikan(self):
        self.selesai = True
        if self.timer_job:
            self.after_cancel(self.timer_job)
        waktu_dipakai = DURASI_UJIAN_DETIK - self.sisa_detik
        self.on_selesai(
            nama=self.nama,
            nim=self.nim,
            kelas=self.kelas,
            jawaban=self.jawaban,
            ditandai=self.ditandai,
            waktu_detik=waktu_dipakai
        )


# ═══════════════════════════════════════════════════════════════════
# LAYAR: REKAP HASIL
# ═══════════════════════════════════════════════════════════════════

class LayarRekap(tk.Frame):
    def __init__(self, master, nama, nim, kelas, jawaban, ditandai,
                 waktu_detik, on_ulang):
        super().__init__(master, bg=CLR["bg_dark"])
        self.nama = nama
        self.nim  = nim
        self.kelas = kelas
        self.jawaban = jawaban
        self.waktu_detik = waktu_detik
        self.on_ulang = on_ulang
        self._hitung_nilai()
        self._bangun()

    def _hitung_nilai(self):
        self.benar  = 0
        self.salah  = 0
        self.kosong = 0
        self.poin_total = 0
        self.detail = []

        for soal in BANK_SOAL:
            jawab = self.jawaban.get(soal.id, None)
            if jawab is None:
                self.kosong += 1
                status = "kosong"
            elif jawab == soal.kunci:
                self.benar += 1
                self.poin_total += soal.poin
                status = "benar"
            else:
                self.salah += 1
                status = "salah"
            self.detail.append((soal, jawab, status))

        total_poin_max = sum(s.poin for s in BANK_SOAL)
        self.nilai = (self.poin_total / total_poin_max) * 100

        if self.nilai >= 90:
            self.grade, self.grade_color = "A", CLR["success"]
        elif self.nilai >= 80:
            self.grade, self.grade_color = "B", CLR["teal"]
        elif self.nilai >= 70:
            self.grade, self.grade_color = "C", CLR["gold"]
        elif self.nilai >= 60:
            self.grade, self.grade_color = "D", CLR["warning"]
        else:
            self.grade, self.grade_color = "E", CLR["danger"]

    def _bangun(self):
        self.pack(fill="both", expand=True)

        # Header
        hdr = tk.Frame(self, bg=CLR["bg_panel"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="  HASIL UJIAN",
                 font=("Consolas", 10, "bold"),
                 bg=CLR["bg_panel"], fg=CLR["accent"]).pack(side="left", pady=12)
        tk.Label(hdr, text=f"  {self.nama}  |  {self.nim}  |  {self.kelas}  ",
                 font=("Segoe UI", 9),
                 bg=CLR["bg_panel"], fg=CLR["text_dim"]).pack(side="right")
        tk.Frame(self, bg=CLR["border"], height=1).pack(fill="x")

        # Scroll container
        container = tk.Frame(self, bg=CLR["bg_dark"])
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, bg=CLR["bg_dark"], highlightthickness=0)
        scroll = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=CLR["bg_dark"])

        self.scroll_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)

        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        canvas.bind_all("<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        self._bangun_konten()

    def _bangun_konten(self):
        pad = {"padx": 24, "pady": 8}
        f = self.scroll_frame

        # ── Panel Nilai Utama ──
        panel_nilai = tk.Frame(f, bg=CLR["bg_panel"])
        panel_nilai.pack(fill="x", padx=24, pady=16)
        tk.Frame(panel_nilai, bg=self.grade_color, height=4).pack(fill="x")

        inner = tk.Frame(panel_nilai, bg=CLR["bg_panel"], pady=20, padx=30)
        inner.pack(fill="x")

        # Grade besar
        lbl_grade = tk.Label(inner, text=self.grade,
                              font=("Consolas", 64, "bold"),
                              bg=CLR["bg_panel"], fg=self.grade_color)
        lbl_grade.pack(side="left")

        # Detail nilai
        detail_frame = tk.Frame(inner, bg=CLR["bg_panel"], padx=20)
        detail_frame.pack(side="left", fill="y", expand=True)

        tk.Label(detail_frame,
                 text=f"{self.nilai:.1f} / 100",
                 font=("Consolas", 28, "bold"),
                 bg=CLR["bg_panel"],
                 fg=CLR["text_bright"]).pack(anchor="w")

        menit = self.waktu_detik // 60
        detik = self.waktu_detik % 60
        info_items = [
            (f"✅ Benar : {self.benar} soal   "
             f"❌ Salah : {self.salah} soal   "
             f"⬜ Kosong: {self.kosong} soal",
             CLR["text_dim"]),
            (f"⏱ Waktu pengerjaan: {menit} menit {detik} detik",
             CLR["text_dim"]),
            (f"📅 {NAMA_UJIAN}  •  {KODE_MK}", CLR["text_muted"]),
        ]
        for teks, warna in info_items:
            tk.Label(detail_frame, text=teks,
                     font=("Segoe UI", 9),
                     bg=CLR["bg_panel"], fg=warna).pack(anchor="w", pady=1)

        # Pesan motivasi
        pesan_map = {
            "A": "🏆 Luar biasa! Anda sangat menguasai materi Bab 1!",
            "B": "🎉 Bagus! Pemahaman Anda sudah baik.",
            "C": "👍 Cukup baik. Pelajari kembali bagian yang salah.",
            "D": "📖 Perlu lebih banyak latihan. Jangan menyerah!",
            "E": "💪 Semangat! Baca kembali materi dan coba lagi.",
        }
        pesan_frame = tk.Frame(inner, bg=CLR["bg_card"],
                                padx=16, pady=10)
        pesan_frame.pack(side="right", fill="y", padx=(20, 0))
        tk.Label(pesan_frame,
                 text=pesan_map[self.grade],
                 font=("Segoe UI", 10),
                 bg=CLR["bg_card"], fg=CLR["text_main"],
                 wraplength=200).pack()

        # ── Statistik per Bagian ──
        tk.Label(f, text="STATISTIK PER BAGIAN",
                 font=("Consolas", 9, "bold"),
                 bg=CLR["bg_dark"], fg=CLR["text_muted"],
                 anchor="w").pack(fill="x", padx=24, pady=(8, 4))

        stat_row = tk.Frame(f, bg=CLR["bg_dark"])
        stat_row.pack(fill="x", padx=24, pady=(0, 12))

        bagian_stats = {}
        for soal, jawab, status in self.detail:
            b = soal.bagian
            if b not in bagian_stats:
                bagian_stats[b] = {"benar": 0, "total": 0, "poin": 0,
                                    "poin_max": 0, "nama": b}
            bagian_stats[b]["total"] += 1
            bagian_stats[b]["poin_max"] += soal.poin
            if status == "benar":
                bagian_stats[b]["benar"] += 1
                bagian_stats[b]["poin"] += soal.poin

        nama_bagian = {
            "A": "Himpunan Bilangan",
            "B": "Nilai Mutlak",
            "C": "Pertidaksamaan",
            "D": "Aplikasi & UTS"
        }
        for b, stat in sorted(bagian_stats.items()):
            pct = stat["poin"] / stat["poin_max"] * 100 if stat["poin_max"] else 0
            warna = self.WARNA_BAGIAN_RKP[b]

            card = tk.Frame(stat_row, bg=CLR["bg_panel"],
                             relief="flat", bd=0)
            card.pack(side="left", fill="y", expand=True, padx=4)
            tk.Frame(card, bg=warna, height=3).pack(fill="x")

            inner2 = tk.Frame(card, bg=CLR["bg_panel"], padx=14, pady=12)
            inner2.pack(fill="x")
            tk.Label(inner2, text=f"Bagian {b}",
                     font=("Consolas", 9, "bold"),
                     bg=CLR["bg_panel"], fg=warna).pack(anchor="w")
            tk.Label(inner2, text=nama_bagian.get(b, ""),
                     font=("Segoe UI", 8),
                     bg=CLR["bg_panel"], fg=CLR["text_dim"]).pack(anchor="w")
            tk.Label(inner2,
                     text=f"{stat['benar']}/{stat['total']} benar",
                     font=("Segoe UI", 9),
                     bg=CLR["bg_panel"], fg=CLR["text_main"]).pack(anchor="w", pady=(4, 0))
            tk.Label(inner2, text=f"{pct:.0f}%",
                     font=("Consolas", 18, "bold"),
                     bg=CLR["bg_panel"], fg=warna).pack(anchor="w")

        # ── Pembahasan Detail ──
        tk.Label(f, text="PEMBAHASAN DETAIL",
                 font=("Consolas", 9, "bold"),
                 bg=CLR["bg_dark"], fg=CLR["text_muted"],
                 anchor="w").pack(fill="x", padx=24, pady=(16, 6))

        for soal, jawab, status in self.detail:
            self._buat_kartu_soal(f, soal, jawab, status)

        # ── Tombol ──
        btn_frame = tk.Frame(f, bg=CLR["bg_dark"], pady=20)
        btn_frame.pack(fill="x", padx=24)

        tk.Button(btn_frame,
                  text="🔁 Ulangi Ujian",
                  font=("Consolas", 10, "bold"),
                  bg=CLR["accent"], fg="white",
                  activebackground=CLR["accent_glow"],
                  relief="flat", bd=0, cursor="hand2",
                  command=self.on_ulang,
                  padx=24, pady=12).pack(side="left", padx=(0, 10))

        tk.Button(btn_frame,
                  text="✖ Keluar",
                  font=("Consolas", 10, "bold"),
                  bg=CLR["bg_panel"], fg=CLR["text_dim"],
                  activebackground=CLR["bg_hover"],
                  relief="flat", bd=0, cursor="hand2",
                  command=lambda: self.master.quit(),
                  padx=24, pady=12).pack(side="left")

    WARNA_BAGIAN_RKP = {
        "A": CLR["accent"],
        "B": CLR["teal"],
        "C": CLR["purple"],
        "D": CLR["gold"],
    }

    def _buat_kartu_soal(self, parent, soal, jawab, status):
        if status == "benar":
            warna_strip = CLR["success"]
            ikon = "✅"
        elif status == "salah":
            warna_strip = CLR["danger"]
            ikon = "❌"
        else:
            warna_strip = CLR["text_muted"]
            ikon = "⬜"

        card = tk.Frame(parent, bg=CLR["bg_card"], relief="flat", bd=0)
        card.pack(fill="x", padx=24, pady=4)
        tk.Frame(card, bg=warna_strip, width=4).pack(side="left", fill="y")

        body = tk.Frame(card, bg=CLR["bg_card"], padx=16, pady=12)
        body.pack(side="left", fill="both", expand=True)

        # Header
        hdr = tk.Frame(body, bg=CLR["bg_card"])
        hdr.pack(fill="x")
        tk.Label(hdr, text=f"{ikon} Soal {soal.id} — {soal.topik}",
                 font=("Consolas", 9, "bold"),
                 bg=CLR["bg_card"], fg=CLR["text_main"]).pack(side="left")
        tk.Label(hdr, text=f"Bagian {soal.bagian}  |  {soal.poin} poin",
                 font=("Segoe UI", 8),
                 bg=CLR["bg_card"], fg=CLR["text_dim"]).pack(side="right")

        # Pertanyaan
        tk.Label(body, text=soal.pertanyaan,
                 font=("Segoe UI", 10),
                 bg=CLR["bg_card"], fg=CLR["text_dim"],
                 wraplength=700, anchor="w", justify="left").pack(fill="x", pady=(4, 8))

        # Pilihan (tampilkan semua, highlight jawaban peserta & kunci)
        for i, (huruf, teks) in enumerate(zip(["A", "B", "C", "D"], soal.pilihan)):
            is_kunci = (huruf == soal.kunci)
            is_dipilih = (huruf == jawab)

            if is_kunci and is_dipilih:
                bg = CLR["success"]
                fg = "white"
                prefix = "✅ "
            elif is_kunci and not is_dipilih:
                bg = "#1a3a1a"
                fg = CLR["success"]
                prefix = "✔ "
            elif is_dipilih and not is_kunci:
                bg = "#3a1a1a"
                fg = CLR["danger"]
                prefix = "✗ "
            else:
                bg = CLR["bg_card"]
                fg = CLR["text_muted"]
                prefix = "   "

            row = tk.Frame(body, bg=bg, padx=8, pady=3)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{prefix}{teks}",
                     font=("Segoe UI", 9),
                     bg=bg, fg=fg,
                     anchor="w", wraplength=680).pack(fill="x")

        # Penjelasan — toggle
        penj_visible = tk.BooleanVar(value=False)
        penj_frame = tk.Frame(body, bg=CLR["bg_card"])

        def toggle_penjelasan():
            if penj_visible.get():
                penj_frame.pack_forget()
                penj_visible.set(False)
                btn_penj.config(text="▶ Lihat Pembahasan")
            else:
                penj_frame.pack(fill="x", pady=(4, 0))
                penj_visible.set(True)
                btn_penj.config(text="▼ Sembunyikan Pembahasan")

        btn_penj = tk.Button(body,
                              text="▶ Lihat Pembahasan",
                              font=("Segoe UI", 8),
                              bg=CLR["bg_card"], fg=CLR["accent"],
                              activebackground=CLR["bg_input"],
                              relief="flat", bd=0, cursor="hand2",
                              command=toggle_penjelasan,
                              anchor="w")
        btn_penj.pack(anchor="w", pady=(6, 0))

        tk.Frame(penj_frame, bg=CLR["bg_input"], padx=12, pady=10).pack(
            fill="x"
        )
        inner_penj = body.children.get(list(body.children.keys())[-1]) or penj_frame

        penj_text = tk.Text(penj_frame,
                             font=("Consolas", 9),
                             bg=CLR["bg_input"],
                             fg=CLR["text_main"],
                             relief="flat", bd=0,
                             wrap="word",
                             state="normal",
                             height=max(4, soal.penjelasan.count("\n") + 3),
                             padx=12, pady=8)
        penj_text.insert("end", soal.penjelasan)
        penj_text.config(state="disabled", cursor="arrow")
        penj_text.pack(fill="x")


# ═══════════════════════════════════════════════════════════════════
# APLIKASI UTAMA
# ═══════════════════════════════════════════════════════════════════

class AplikasiCAT:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"CAT — {NAMA_UJIAN} | {INSTITUSI}")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)
        self.root.configure(bg=CLR["bg_dark"])

        # Icon (emoji sebagai fallback)
        try:
            self.root.iconbitmap("icon.ico")
        except Exception:
            pass

        self._layar_sekarang = None
        self._ke_login()

    def _bersihkan(self):
        if self._layar_sekarang:
            self._layar_sekarang.destroy()
            self._layar_sekarang = None

    def _ke_login(self):
        self._bersihkan()
        self._layar_sekarang = LayarLogin(self.root, on_mulai=self._ke_ujian)

    def _ke_ujian(self, nama, nim, kelas):
        self._bersihkan()
        self._layar_sekarang = LayarUjian(
            self.root, nama=nama, nim=nim, kelas=kelas,
            on_selesai=self._ke_rekap
        )

    def _ke_rekap(self, nama, nim, kelas, jawaban, ditandai, waktu_detik):
        self._bersihkan()
        self._layar_sekarang = LayarRekap(
            self.root,
            nama=nama, nim=nim, kelas=kelas,
            jawaban=jawaban, ditandai=ditandai,
            waktu_detik=waktu_detik,
            on_ulang=self._ke_login
        )

    def jalankan(self):
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = AplikasiCAT()
    app.jalankan()