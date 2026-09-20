# HoloMine — Property Price Prediction from Sales Description (Task 2)

Prediksi `listPrice` dari teks listing properti. Metrik: **MAE (dolar)**.

| Data | Baris | Kolom |
|---|---|---|
| `train.csv` | 14.640 | `id`, `text`, `listPrice` |
| `test.csv` | 3.659 | `id`, `text` |

Harga sangat *heavy-tailed*: min \$1, median \$499.900, maks \$80.000.000.
Menebak konstanta (median) memberi MAE ≈ **550.252** — itu baseline yang harus dikalahkan.

---

## 1. Keputusan desain yang paling menentukan skor

**Latih di ruang log, prediksi median.**
MAE diminimalkan oleh **median bersyarat**, bukan rata-rata. Karena `exp()` monoton,
median di ruang log = log dari median di ruang dolar. Jadi:

```
target  = log(listPrice)
loss    = L1 / Huber / epsilon-insensitive   (bukan MSE)
prediksi = exp(output)
```

Ini sekaligus mencegah satu rumah \$80 juta mendominasi gradien. Melatih langsung
di skala dolar dengan MSE adalah kesalahan paling mahal di kompetisi ini.

**Validasi**: 5-fold `StratifiedKFold` atas **desil log-harga** (`common.price_bin_folds`),
supaya ekor mahal terbagi rata di setiap fold. Semua model memakai fold yang sama
sehingga OOF-nya bisa di-blend dengan jujur.

---

## 2. Hasil CV yang sudah terukur (5-fold, MAE dolar)

Semua angka di bawah dijalankan di repo ini dengan fold yang sama.

| Model | CV MAE | Catatan |
|---|---:|---|
| Tebak konstanta (median) | 550.252 | baseline |
| Ridge TF-IDF char 3-5 | 362.280 | |
| LinearSVR TF-IDF kata | 359.217 | L1 -> median |
| Ridge TF-IDF kata 1-2 | 358.966 | |
| LightGBM (SVD-250 + fitur regex) | 357.169 | |
| **kNN kosinus TF-IDF (median tetangga)** | **353.974** | model tunggal terbaik |
| Greedy blend (rata-rata berbobot) | 337.639 | |
| + stacking LightGBM level-2 | 314.247 | lompatan terbesar |
| + kalibrasi per-desil | **313.478** | `submissions/submission_v2_sparse.csv` |

**43% lebih baik dari baseline, tanpa GPU sama sekali.** Stacking memberi lompatan
terbesar karena model level-2 belajar *di mana* tiap model bisa dipercaya
(kNN kuat saat ada listing kembar, ridge kuat untuk nama kota langka) sambil
tetap melihat fitur numerik hasil ekstraksi regex.

Fine-tune transformer belum bisa dijalankan di sini (akses ke huggingface.co
diblokir oleh policy jaringan lingkungan ini), tetapi seluruh pipeline-nya sudah
ditulis dan diuji end-to-end memakai encoder kecil yang dilatih dari nol --
loop training, pooling, layer-wise LR, penyimpanan OOF semuanya berjalan.
Jalankan di Kaggle/Colab dengan GPU untuk memakainya.

---

## 3. Struktur

**Semua kode ada di satu file: `holomine_solution.py`.**

```
holomine_solution.py             SELURUH pipeline, satu file tanpa dependensi internal
  bagian 1  konfigurasi + daftar model HuggingFace yang direkomendasikan
  bagian 2  data, fold, metrik, transformasi target
  bagian 3  information extraction regex -> ~80 fitur numerik
  bagian 4  model CPU: TF-IDF ridge/SVR, kNN kosinus, LightGBM
  bagian 5  embedding beku (sentence-transformers)
  bagian 6  fine-tune encoder HuggingFace
  bagian 7  blend: greedy selection + stacking + kalibrasi
  bagian 8  runner
notebooks/holomine_kaggle.ipynb  notebook Kaggle siap Run All
artifacts/                       oof_<model>.npy & test_<model>.npy (ruang log)
submissions/                     file siap unggah ke Kaggle
```

Setiap model menulis OOF dengan format yang sama, jadi tahap blend otomatis
memungut model baru apa pun yang sudah dilatih.

---

## 4. Output: di mana file submission-nya

Skrip menulis `submission.csv` dengan format persis `sample_submission.csv`
(kolom `id` + `listPrice`, 3.659 baris, urutan id sama), hanya saja kolom
`listPrice` yang tadinya `0` sudah berisi prediksi:

```
id,listPrice
RE_015637,427069.05
RE_005176,452326.59
```

Lokasinya:

| Dijalankan di | File submission |
|---|---|
| lokal (skrip) | `submissions/submission.csv` |
| Kaggle / Colab | `/kaggle/working/submission.csv` |

Ubah nama file dengan `--out namaku.csv` atau `run(out="namaku.csv")`.
Setiap tahap diakhiri blend, jadi `run(stage="cpu")` pun tetap menghasilkan
submission -- bukan cuma artefak `.npy`. Baris terakhir di log selalu mencetak
path lengkapnya.

---

## 5. Cara menjalankan

### Satu perintah (disarankan)

```bash
pip install -r requirements.txt

python holomine_solution.py --dry-run     # lihat rencananya dulu
python holomine_solution.py               # jalankan kombinasi inti
python holomine_solution.py --tier extra  # + model besar kalau kuota GPU masih ada
```

Skrip ini menjalankan berurutan: model CPU -> submission sementara ->
embedding beku -> fine-tune -> blend akhir. Tahap yang artefaknya sudah ada
**dilewati**, jadi aman dijalankan ulang setelah sesi Kaggle terputus. Tahap yang
gagal dicatat dan tidak menghentikan tahap lain, sehingga Anda selalu punya
submission yang valid. Tanpa GPU, tahap fine-tune otomatis dilewati.

### Dari dalam notebook (Kaggle/Colab)

Kalau isi file ditempel ke sel notebook, **jangan ubah `ROOT`** — skrip sudah
menangani sendiri tidak adanya `__file__`, mencari `train.csv` otomatis (termasuk
`/kaggle/input/<kompetisi>/` maupun `/kaggle/input/competitions/<kompetisi>/`),
dan menulis output ke `/kaggle/working`. Panggil `run()`, bukan argumen CLI:

```python
run()                  # jalankan semua
run(stage="cpu")       # model CPU saja
run(tier="extra")      # + model besar
run(folds="0,1")       # uji cepat 2 fold
```

`argparse` sengaja tidak membaca `sys.argv` di notebook, karena isinya milik
kernel (`-f /tmp/xxx.json ...`) dan akan membuatnya mati dengan `SystemExit: 2`.

Kalau data tetap tidak ketemu, paksa lokasinya:

```python
import os; os.environ["HOLOMINE_DATA"] = "/kaggle/input/nama-kompetisi"
```

Cara paling rapi tetap menjalankannya sebagai skrip:
`!python holomine_solution.py --tier core`, atau upload
`notebooks/holomine_kaggle.ipynb` lalu Run All (Accelerator GPU, Internet ON).

### Per tahap (kalau mau kontrol penuh)

```bash
python holomine_solution.py --stage cpu        # model CPU saja, tanpa GPU (~10 menit)
python holomine_solution.py --stage embed      # embedding beku saja
python holomine_solution.py --stage finetune   # fine-tune saja
python holomine_solution.py --stage blend      # blend ulang dari artefak yang ada
```

Opsi lain: `--folds 0,1` (uji cepat), `--force` (latih ulang),
`--precision fp32` (kalau fp16 divergen di deberta-v3-large).
Semua hyperparameter per-model ada di dict `FINETUNE` / `EMBED` di bagian 1 file.

---

## 6. Kombinasi model HuggingFace yang direkomendasikan

Teks listing berbahasa Inggris, median ~870 karakter (~171 token, p99 521 token),
dan sinyal harganya tersebar: lokasi, luas, jumlah kamar, kondisi, serta kata-kata
mewah. Yang dibutuhkan encoder pemahaman teks — bukan LLM generatif.

### Prioritas 1 — fine-tune (kontribusi terbesar)

| Model | Kenapa cocok | Catatan |
|---|---|---|
| [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base) | Juara de-facto regresi teks di Kaggle (disentangled attention). Titik awal terbaik. | `max_len 512`, lr 2e-5, 3 epoch |
| [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) | Arsitektur & tokenizer 2024 yang berbeda dari DeBERTa, dan lebih cepat. | `max_len 512`, butuh `transformers>=4.48` |
| [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large) | Biasanya 2–4% lebih baik dari base bila tuning stabil. | lr 8e-6, `--grad-checkpoint`, batch kecil |
| [`answerdotai/ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) | Pasangan besar yang beragam dari DeBERTa. | perlu ≥16 GB VRAM |

**Soal panjang teks:** diukur langsung di `train.csv`, p99 = 521 token dan pada
`max_len=512` hanya **1,1%** listing yang terpotong (itu pun dengan tokenizer
bervocab kecil yang memecah kata lebih banyak daripada tokenizer asli kedua model
ini). Jadi konteks panjang ModernBERT **bukan** keunggulan nyata di sini — semua
model dijalankan di 512, dan nilai ModernBERT murni dari keberagaman arsitektur.
Ini menghemat separuh waktu latihnya.

### Prioritas 2 — embedding beku (murah, menambah keberagaman)

| Model | Kenapa |
|---|---|
| [`Alibaba-NLP/gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | Embedding kuat berbasis ModernBERT, 8k konteks |
| [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) / [`bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | Standar industri; versi small enak untuk CPU |
| [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Peringkat MTEB teratas di kelas kecil; dimensi besar |
| [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2) | Perlu prefiks `"query: "` (`--prefix "query: "`) |
| [`ibm-granite/granite-embedding-english-r2`](https://huggingface.co/ibm-granite/granite-embedding-english-r2) | Rilis 2025/2026, khusus Inggris |
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Tercepat; untuk iterasi di CPU |

### Kombinasi final yang disarankan

```
DeBERTa-v3-base (fine-tune)         pemahaman semantik terbaik
+ ModernBERT-base (fine-tune)       arsitektur/tokenizer berbeda -> error tidak berkorelasi
+ gte-modernbert-base (beku + LGBM) sudut pandang berbeda, biaya rendah
+ kNN kosinus TF-IDF                menangkap listing "kembar" di pasar yang sama
+ LightGBM(SVD + fitur regex)       menangkap angka eksplisit: sqft, kamar, acre
+ Ridge / LinearSVR TF-IDF          menangkap nama kota & kata kunci langka
```

Persis inilah yang dijalankan `python holomine_solution.py` (`--tier core`).
Estimasi waktu di satu T4: ~10 menit tahap CPU, ~10 menit embedding,
~30-45 menit per model fine-tune untuk 5 fold.

Perkiraan: menambahkan DeBERTa-v3-base + ModernBERT-base ke blend ini biasanya
memangkas MAE cukup besar lagi, karena keduanya memahami konteks kalimat yang
tidak bisa ditangkap TF-IDF (mis. "butuh renovasi total" vs "baru direnovasi
total" -- bag-of-words melihat kata yang hampir sama, transformer tidak).

Keberagaman lebih berharga daripada satu model terkuat: blend digabung dengan
*greedy ensemble selection* langsung terhadap MAE, jadi model yang lemah tapi
berbeda tetap bisa terpilih, dan model yang tidak membantu otomatis dibuang.

### Yang TIDAK disarankan
- LLM generatif (Llama/Mistral/Qwen-Instruct) untuk memprediksi angka sebagai teks —
  mahal dan kalah dari encoder+head regresi pada data sekecil ini.
- Model multibahasa (XLM-R) — datanya murni Inggris, kapasitasnya terbuang.
- MSE di skala dolar — lihat bagian 1.
