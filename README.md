# HoloMine Task 2 — Prediksi Harga Properti dari Teks Listing

Satu file: **`train_and_predict.py`**. Melatih `train.csv`, memprediksi `test.csv`,
lalu mengisi kolom `listPrice` di `sample_submission.csv` → `submission.csv`.

```bash
pip install numpy pandas scikit-learn scipy lightgbm
pip install torch transformers sentencepiece      # untuk tahap GPU

python train_and_predict.py                  # semua (butuh GPU T4)
python train_and_predict.py --quick          # 1 fold 1 epoch, uji cepat
python train_and_predict.py --no-transformer # CPU saja, ~8 menit
```

Di Kaggle: Settings → Accelerator **GPU T4**, Internet **ON**, lalu satu sel:

```python
!python train_and_predict.py
```

Submission mendarat di `/kaggle/working/submission.csv` — folder itulah yang
menjadi **Output** notebook setelah Save Version.

---

## Yang dikerjakan skrip, berurutan

| Tahap | Isi | Waktu |
|---|---|---|
| 1 | Baca train.csv (14.640) + test.csv (3.659) | detik |
| 2 | **Latih** 5 model CPU: ridge kata, LinearSVR, kNN kosinus, ridge karakter, LightGBM | ~8 menit |
| 3 | **Latih** DeBERTa-v3-base di GPU T4, 5 fold | ~60 menit |
| 4 | Gabungkan: greedy blend → stacking → kalibrasi | ~2 menit |
| 5 | Tulis submission.csv | detik |

Tidak ada cache dan tidak ada tahap yang bisa dilewati: sekali jalan = semuanya
dilatih ulang.

---

## Keputusan yang paling menentukan skor

**Target `log(listPrice)`, loss L1/Huber, lalu `exp()`.**
MAE diminimalkan oleh **median bersyarat**, bukan rata-rata. Karena `exp()`
monoton, median di ruang log = log dari median di ruang dolar — jadi `exp()` dari
fit L1 di ruang log *persis* yang diminta metrik. Ini sekaligus mencegah satu
rumah $80 juta mendominasi gradien. Melatih di skala dolar dengan MSE adalah
kesalahan paling mahal di lomba ini.

**Validasi:** 5-fold distratifikasi atas desil log-harga, dipakai sama oleh semua
model supaya prediksinya bisa digabung dengan jujur.

---

## Hasil terukur (5-fold, MAE dolar)

| Model | CV MAE |
|---|---:|
| Tebak konstanta (median) | 550.252 |
| Ridge TF-IDF karakter | 362.280 |
| LinearSVR TF-IDF kata | 359.217 |
| Ridge TF-IDF kata | 358.966 |
| LightGBM (SVD + fitur regex) | 357.169 |
| **kNN kosinus (median tetangga)** | **353.974** |
| Rata-rata berbobot (greedy) | 337.639 |
| + stacking LightGBM | 314.247 |
| + kalibrasi per-desil | **313.478** |

43% lebih baik dari baseline, **belum termasuk transformer** (belum terukur —
lihat catatan di bawah). Lompatan terbesar dari stacking: model level-2 melihat
prediksi tiap model *bersama* fitur numerik hasil regex, jadi ia belajar **di
mana** tiap model bisa dipercaya, bukan memberi satu bobot global.

### Catatan kejujuran

Angka transformer **belum pernah diukur**: lingkungan tempat kode ini ditulis
memblokir `huggingface.co`, jadi bobot DeBERTa asli tidak bisa diunduh. Yang
sudah diuji: seluruh loop training-nya (memakai encoder kecil yang dilatih dari
nol — hasilnya mengalahkan baseline, jadi mekanismenya terbukti jalan), jalur
CPU penuh, penulisan submission, dan skenario notebook Kaggle. Angka aslinya
baru muncul saat Anda menjalankannya di T4.

---

## Kenapa model-modelnya itu

| Model | Menangkap apa |
|---|---|
| Ridge TF-IDF kata | nama kota & kata kunci langka |
| Ridge TF-IDF karakter | variasi penulisan & salah ketik |
| LinearSVR | sama seperti ridge tapi loss L1 → median |
| kNN kosinus | listing "kembar" di pasar yang sama |
| LightGBM + regex | angka eksplisit: sqft, kamar, acre, tahun bangun |
| DeBERTa-v3-base | makna kalimat — mis. "butuh renovasi total" vs "baru direnovasi total", yang bagi bag-of-words nyaris sama |

Ekstraksi regex memulihkan ~80 fitur numerik dari prosa (kamar, kamar mandi,
luas, tanah, tahun, garasi, HOA) plus ~40 sinyal fasilitas/kondisi — kolom yang
biasanya dimiliki model properti tapi di sini harus digali dari teks.

## Opsi lain

```
--model answerdotai/ModernBERT-base   ganti backbone
--folds 3                             lebih sedikit fold (lebih cepat)
--epochs 2 --batch-size 8             hemat VRAM/waktu
--no-cpu-models                       transformer saja
--out namaku.csv                      ganti nama file
```
