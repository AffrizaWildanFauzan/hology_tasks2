# HoloMine Task 2 — Prediksi Harga Properti dari Teks Listing

Satu file: **`train_and_predict.py`**. Melatih `train.csv`, memprediksi `test.csv`,
lalu mengisi kolom `listPrice` di `sample_submission.csv` → `submission.csv`.

```bash
pip install numpy pandas scikit-learn scipy lightgbm
pip install torch transformers sentencepiece      # untuk tahap GPU

python train_and_predict.py                  # ~30 menit total di T4
python train_and_predict.py --quick          # uji cepat
python train_and_predict.py --no-transformer # CPU saja, ~10 menit

# kalau waktu longgar dan mau akurasi maksimal (~2 jam):
python train_and_predict.py --model microsoft/deberta-v3-base --batch-size 16 --folds 5
```

**Default sengaja ringan.** Backbone-nya `deberta-v3-xsmall` (22M parameter,
~5x lebih cepat dari base) dengan 3 fold. `--folds` adalah jumlah *split*:
berapa pun nilainya, seluruh baris tetap mendapat prediksi out-of-fold, jadi
mengecilkannya menghemat waktu tanpa membuang model itu dari blend.

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
| 2 | **Latih** 6 model CPU: ridge kata/karakter, LinearSVR, kNN kata/karakter, LightGBM | ~10 menit |
| 3 | **Latih** DeBERTa-v3-**xsmall** di GPU T4, 3 fold | ~20 menit; skrip mencetak estimasinya sendiri setelah epoch pertama |
| 4 | Gabungkan: greedy blend → stacking → kalibrasi | ~2 menit |
| 5 | Tulis submission.csv | detik |

Tidak ada cache dan tidak ada tahap yang bisa dilewati: sekali jalan = semuanya
dilatih ulang.

---

## Catatan presisi di GPU T4

T4 adalah Turing (sm_75) dan **tidak punya bf16 native**, tapi
`torch.cuda.is_bf16_supported()` di PyTorch baru tetap mengembalikan `True`
karena menghitung bf16 emulasi — dan emulasinya menghasilkan NaN. Karena itu
presisi dipilih dari compute capability (`sm_80+` baru bf16), bukan dari fungsi
itu.

Bobot juga dipaksa fp32 saat dimuat. `transformers` versi baru memuat checkpoint
dengan dtype aslinya; kalau itu fp16, gradiennya ikut fp16 dan `GradScaler`
melempar *"Attempting to unscale FP16 gradients"*. Mixed precision yang benar
adalah **bobot fp32 + autocast**, bukan bobot fp16.

Kalau training tetap meledak, skrip mendeteksinya dalam 30 langkah pertama lalu
mengulang sendiri di fp32 — jadi kegagalan presisi memakan hitungan detik, bukan
satu jam. Kalau fp32 pun gagal, tahap transformer dilewati dan submission tetap
dibuat dari model CPU.

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
| kNN karakter | 365.574 |
| Ridge TF-IDF karakter | 362.280 |
| LinearSVR TF-IDF kata | 359.217 |
| Ridge TF-IDF kata | 358.966 |
| LightGBM (SVD + regex + uniqueness) | 356.962 |
| **kNN kosinus kata (median tetangga)** | **353.974** |
| Rata-rata berbobot (greedy) | 337.508 |
| + stacking LightGBM | 310.358 |
| + kalibrasi per-desil | **309.622** |

44% lebih baik dari baseline, **belum termasuk transformer** (belum terukur —
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

## Riset: apa yang sudah diuji dari literatur

Literatur sepakat teks itu berharga: Baur dkk. menurunkan MAE valuasi properti
hingga **17%** dengan menambahkan deskripsi ke model terstruktur, Nowak dkk.
melaporkan penurunan error 18,7–39,1%, dan Zhang dkk. menunjukkan deskripsi
**saja** sudah prediktor kuat (R² 0,79). Itu persis situasi lomba ini — hanya
teks yang tersedia.

Dua ide dari paper saya uji langsung di data ini, dengan fold yang sama:

| Ide | Sumber | Hasil terukur | Putusan |
|---|---|---|---|
| Skor **uniqueness** deskripsi | Shen dkk., *J. Urban Economics* 2021 | sendirian nyaris nol (korelasi −0,003 dengan log-harga, MAE −207), tapi **−2.767 di dalam blend** (CI95 [−4.149, −1.447]) | **dipakai** |
| **Regression-via-classification** (diskretkan harga jadi 24 bin, ambil median distribusi) | Díaz & Marathe 2019; Berg dkk. 2020; Shah dkk. 2022 | 383.277 sendirian (jauh lebih buruk), +199 di blend | **ditolak** |
| kNN di ruang **karakter** | — (ide sendiri, dari keberagaman) | 365.574 sendirian, **−1.933 di blend** (CI95 [−6.608, −2.543] kumulatif) | **dipakai** |
| kNN karakter versi **SVD** (50x lebih cepat) | — | +622 di blend | **ditolak** |
| kNN dengan k=8 dan k=75 | — | +1.229 di blend | **ditolak** |

Pelajarannya: **kualitas satu model bukan ukuran nilainya di dalam ensemble.**
Fitur uniqueness hampir tidak berguna sendirian tapi jelas membantu blend,
sementara RvC yang punya dasar teori kuat justru tidak menambah apa-apa di sini.
Semua klaim di atas diuji dengan *paired bootstrap* 2.000x — selisih di bawah
~1.500 MAE tidak bisa dibedakan dari derau pada dataset sebesar ini, jadi
perbaikan yang lebih kecil dari itu saya buang.

### Yang belum diuji (butuh GPU + akses HuggingFace)

Target distribusi harga di sini sangat miring ($1 sampai $80 juta), dan ada
literatur khusus untuk itu — **deep imbalanced regression**: label/feature
distribution smoothing (Yang dkk. 2021), Balanced MSE (Ren dkk. 2022), dan
RankSim (Gong dkk. 2022). Semuanya menyasar persis masalah ekor panjang ini dan
layak dicoba pada kepala regresi transformer. Begitu pula kepala **soft-label**
di atas bin harga, yang gagal untuk LightGBM tapi justru paling sering berhasil
untuk jaringan dalam — itulah konteks asal papernya.

---

## Kenapa model-modelnya itu

| Model | Menangkap apa |
|---|---|
| Ridge TF-IDF kata | nama kota & kata kunci langka |
| Ridge TF-IDF karakter | variasi penulisan & salah ketik |
| LinearSVR | sama seperti ridge tapi loss L1 → median |
| kNN kosinus kata | listing "kembar" di pasar yang sama |
| kNN kosinus karakter | tetangga yang luput dari kemiripan kata |
| Skor uniqueness | seberapa tidak biasa deskripsinya dibanding pasar |
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
--precision fp16                     paksa presisi (auto sudah benar untuk T4)
--no-cpu-models                       transformer saja
--out namaku.csv                      ganti nama file
```
