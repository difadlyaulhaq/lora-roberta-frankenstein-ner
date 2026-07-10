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
│   └── evaluate.py        # Modul kalkulasi metrik (Linguistik, Komputasi, MUC-5, Fine-grained)
├── requirements.txt       # Daftar dependensi Python
└── README.md              # Dokumentasi proyek
```
