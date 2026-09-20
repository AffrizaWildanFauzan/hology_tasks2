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

## 2. Struktur

```
src/common.py             loader, fold, metrik, transformasi target
src/features.py           information extraction dari teks (regex) -> ~80 fitur numerik
src/run_sparse.py         model CPU: TF-IDF ridge/SVR, kNN kosinus, LightGBM
src/embed_features.py     embedding beku (sentence-transformers) -> Ridge/LightGBM
src/train_transformer.py  fine-tune encoder HuggingFace (GPU)
src/blend.py              greedy ensemble selection + kalibrasi -> submission
artifacts/                oof_<model>.npy & test_<model>.npy (semua di ruang log)
submissions/              file siap unggah ke Kaggle
```

Setiap model menulis OOF dengan format yang sama, jadi `blend.py` otomatis
memungut model baru apa pun yang sudah dilatih.

---

## 3. Cara menjalankan

```bash
pip install -r requirements.txt

# A. Pipeline CPU (tanpa GPU, ~30 menit)
python src/run_sparse.py --models ridge_word,ridge_char,linsvr_word,knn_word,lgbm_dense,lgbm_sparse

# B. Embedding beku (CPU bisa, GPU jauh lebih cepat)
python src/embed_features.py --model BAAI/bge-small-en-v1.5 --tag bge --with-features

# C. Fine-tune transformer (GPU)
python src/train_transformer.py --model microsoft/deberta-v3-base   --tag deb3base --bf16
python src/train_transformer.py --model answerdotai/ModernBERT-base --tag mbert --max-len 1024 --bf16

# D. Gabungkan semuanya
python src/blend.py --calibrate --out submission_blend.csv
```

---

## 4. Kombinasi model HuggingFace yang direkomendasikan

Teks listing panjang (median ~870 karakter, maks ~4.000 ≈ 900 token), berbahasa
Inggris, dan sinyal harganya tersebar: lokasi, luas, jumlah kamar, kondisi, dan
kata-kata mewah. Yang dibutuhkan encoder yang kuat di teks panjang — bukan LLM generatif.

### Prioritas 1 — fine-tune (kontribusi terbesar)

| Model | Kenapa cocok | Catatan |
|---|---|---|
| [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base) | Juara de-facto regresi teks di Kaggle (disentangled attention). Titik awal terbaik. | `max_len 512`, lr 2e-5, 3 epoch |
| [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) | Konteks 8k, arsitektur 2024, cepat. Bisa membaca listing terpanjang **utuh**. | `--max-len 1024`, butuh `transformers>=4.48` |
| [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large) | Biasanya 2–4% lebih baik dari base bila tuning stabil. | lr 8e-6, `--grad-checkpoint`, batch kecil |
| [`answerdotai/ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) | Pasangan besar yang beragam dari DeBERTa. | perlu ≥16 GB VRAM |

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
DeBERTa-v3-base (fine-tune)        bobot besar — pemahaman semantik terbaik
+ ModernBERT-base (fine-tune)      arsitektur/tokenizer berbeda -> error tidak berkorelasi
+ gte-modernbert-base (beku + LGBM) tangkapan sinyal berbeda, biaya rendah
+ kNN kosinus TF-IDF                menangkap listing "kembar" di pasar yang sama
+ LightGBM(SVD + fitur regex)        menangkap angka eksplisit: sqft, kamar, acre
+ Ridge / LinearSVR TF-IDF           menangkap nama kota & kata kunci langka
```

Keberagaman lebih berharga daripada satu model terkuat: blend digabung dengan
*greedy ensemble selection* langsung terhadap MAE, jadi model yang lemah tapi
berbeda tetap bisa terpilih, dan model yang tidak membantu otomatis dibuang.

### Yang TIDAK disarankan
- LLM generatif (Llama/Mistral/Qwen-Instruct) untuk memprediksi angka sebagai teks —
  mahal dan kalah dari encoder+head regresi pada data sekecil ini.
- Model multibahasa (XLM-R) — datanya murni Inggris, kapasitasnya terbuang.
- MSE di skala dolar — lihat bagian 1.
