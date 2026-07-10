# Evaluasi Parameter-Efficient Fine-Tuning (LoRA) pada Model RoBERTa untuk Named Entity Recognition di Domain Sastra Low-Resource

Repositori ini berisi kode dan metodologi eksperimen untuk penelitian evaluasi kinerja model bahasa pra-terlatih (RoBERTa) menggunakan teknik *Low-Rank Adaptation* (LoRA) pada tugas *Named Entity Recognition* (NER). Penelitian ini secara khusus menangani tantangan *domain shift* dan keterbatasan data (*low-resource*) pada teks sastra klasik berbahasa Inggris (novel *Frankenstein*).

## Deskripsi Proyek
Ekstraksi informasi pada teks sastra menghadapi tantangan besar akibat ambiguitas semantik, gaya bahasa kiasan (personifikasi), dan ketiadaan korpus beranotasi yang memadai. Proyek ini mengevaluasi sejauh mana implementasi LoRA pada arsitektur RoBERTa mampu menekan beban komputasi (VRAM) sekaligus bertindak sebagai agen regularisasi intrinsik untuk mencegah *overfitting*, sambil tetap mempertahankan akurasi klasifikasi 18 kelas entitas berdasarkan standar **OntoNotes 5.0**.

## Fitur Utama & Metodologi
* **Model Arsitektur:** RoBERTa (*Robustly Optimized BERT Pretraining Approach*).
* **Teknik Optimasi:** *Parameter-Efficient Fine-Tuning* (PEFT) menggunakan **LoRA**. Pembaruan bobot ($\Delta W = BA$) dibatasi melalui dekomposisi matriks berdimensi rendah untuk mengendalikan derajat kebebasan model.
* **Hyperparameter Tuning:** Menggunakan metode **Grid Search** untuk memetakan kombinasi optimal antara dimensi *Rank* (`r`) dan faktor skala *Alpha* (`alpha`).
* **Dataset Target:** Korpus *low-resource* dari novel *Frankenstein*, dianotasi dalam format BIO (*Beginning, Inside, Outside*).

## Metrik Evaluasi
Kinerja model diuji secara komprehensif menggunakan empat kelompok parameter evaluasi:

1. **Kinerja Linguistik:**
   * Precision, Recall, dan F1-Score (Macro-average).
2. **Kinerja Infrastruktur Komputasi:**
   * Alokasi memori grafis / VRAM maksimal (GB).
   * Waktu komputasi / *Execution Time* (detik/menit).
3. **Analisis Galat (MUC-5 Adaptation):**
   * **COR (Correct):** Entitas dilabeli dengan tepat.
   * **INC (Incorrect):** Kesalahan klasifikasi kelas (*Substitution*).
   * **MIS (Missing):** Gagal mengekstraksi entitas (*Undergeneration*).
   * **SPU (Spurious):** Halusinasi pengekstraksian token non-entitas (*Overgeneration*).
4. **Fine-Grained Analysis (Atribut Entitas):**
   * `eLen` (Entity Length): Ketahanan terhadap frasa entitas panjang ($\ge$ 4 kata).
   * `eCon` (Label Consistency): Penanganan ambiguitas kelas token.
   * `eFre` (Entity Frequency): Kinerja pada entitas *few-shot/zero-shot*.

## Teknologi & Dependensi
Eksperimen ini dibangun menggunakan ekosistem Python dengan pustaka utama berikut:
* `Python` >= 3.8
* `PyTorch` (Backend Komputasi Tensors)
* `Transformers` (Hugging Face - Inisialisasi RoBERTa)
* `PEFT` (Hugging Face - Injeksi LoRA)
* `Datasets` & `Tokenizers`
* `scikit-learn` & `seqeval` (Kalkulasi Metrik Evaluasi)

## Alur Eksekusi (Pipeline)
1. **Pra-pemrosesan Data:** Ekstraksi teks mentah, pelabelan tokenisasi BIO (18 Kelas OntoNotes 5.0), dan penyesuaian *input tensor* menggunakan *tokenizer* RoBERTa.
2. **Setup Arsitektur:** Pembekuan (*freezing*) bobot asli model dan penyisipan matriks LoRA pada lapisan *Transformer*.
3. **Pelatihan (Fine-Tuning):** Menjalankan *Grid Search* iteratif untuk variabel `r` dan `alpha`.
4. **Inferensi & Evaluasi:** Ekstraksi prediksi dari data uji dan kalkulasi metrik performa (F1-Score, VRAM profiler, MUC-5, Fine-grained).

## Struktur Repositori
```text
├── data/                  # Korpus teks mentah dan dataset beranotasi (Frankenstein)
├── notebooks/             # Eksperimen interaktif (Jupyter Notebook / Colab)
├── src/
│   ├── data_loader.py     # Script pra-pemrosesan dan tokenisasi BIO
│   ├── model.py           # Konfigurasi RoBERTa dan integrasi LoRA
│   ├── train.py           # Pipeline Grid Search dan Fine-tuning
│   ├── evaluate.py        # Modul kalkulasi metrik (Linguistik, Komputasi, MUC-5, Fine-grained)
│   └── main.py            # Script utama orkestrator CLI pipeline
├── requirements.txt       # Daftar dependensi Python
└── README.md              # Dokumentasi proyek
```

## 🚀 Cara Menjalankan via CMD / Terminal

Ikuti langkah-langkah di bawah ini untuk menjalankan eksperimen fine-tuning LoRA-RoBERTa menggunakan Command Prompt (CMD) atau PowerShell:

### 1. Instalasi Dependensi
Pastikan Python ($\ge$ 3.8) sudah terinstal, kemudian jalankan perintah berikut untuk menginstal seluruh pustaka yang diperlukan:
```cmd
pip install -r requirements.txt
```

### 2. Jalankan Fine-Tuning Standar
Perintah ini akan menjalankan training standar menggunakan konfigurasi bawaan LoRA ($r=8$, $\alpha=16$) dan langsung mengevaluasi model pada data uji (menyimpan laporan lengkap hasil linguistik, komputasi, MUC-5, dan fine-grained dalam format JSON):
```cmd
python src/main.py --data_path data/frankenstein_annotated.json --output_dir ./results --epochs 3 --batch_size 8
```

### 3. Jalankan Hyperparameter Tuning (Grid Search)
Untuk menyisir performa kombinasi Rank ($r \in [4, 8, 16]$) dan Alpha ($\alpha \in [8, 16, 32]$) guna memetakan performa terbaik (F1-score) serta melacak konsumsi VRAM dan kecepatan latih, jalankan perintah berikut:
```cmd
python src/main.py --data_path data/frankenstein_annotated.json --output_dir ./results --grid_search
```

### ⚙️ Referensi Parameter CLI (CMD Arguments)
Anda dapat menyesuaikan parameter eksekusi dengan menambahkan argumen berikut saat menjalankan script:

| Parameter | Tipe | Nilai Bawaan (Default) | Penjelasan |
| :--- | :--- | :--- | :--- |
| `--data_path` | `str` | `data/frankenstein_annotated.json` | Lokasi file dataset berformat BIO JSON/JSONL/CSV. |
| `--model_name` | `str` | `roberta-base` | Model pre-trained checkpoint dari Hugging Face. |
| `--output_dir` | `str` | `./results` | Folder tempat penyimpanan model terbaik dan file hasil evaluasi. |
| `--grid_search` | `flag` | - | Mengaktifkan sweep parameter LoRA rank dan alpha (menyimpan hasil ke `grid_search_results.csv`). |
| `--epochs` | `int` | `3` | Jumlah siklus pelatihan (hanya berlaku jika tidak memakai `--grid_search`). |
| `--batch_size` | `int` | `8` | Ukuran batch pelatihan untuk input tensor. |
| `--learning_rate`| `float`| `5e-4` | Learning rate yang dioptimalkan untuk LoRA adapter. |

Contoh perintah dengan parameter kustom:
```cmd
python src/main.py --data_path data/frankenstein_annotated.json --epochs 5 --batch_size 16 --learning_rate 3e-4 --output_dir ./custom_results
```

