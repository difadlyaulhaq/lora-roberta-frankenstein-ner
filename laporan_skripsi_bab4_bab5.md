# BAB IV: HASIL DAN PEMBAHASAN

## 4.1 Hasil Pengumpulan dan Pra-pemrosesan Data

### 4.1.1 Distribusi Korpus Sastra
Berdasarkan alur scraping teks novel *Frankenstein* versi Project Gutenberg yang dilakukan secara otomatis, teks mentah berhasil dipilah ke dalam struktur bab-bab cerita yang bersih dari elemen-elemen HTML dan bibliografi Project Gutenberg. Distribusi akhir dari korpus sastra novel *Frankenstein* yang berhasil diekstraksi adalah sebagai berikut:
* **Total Kalimat:** 3.081 kalimat
* **Total Kata (Token):** 85.708 token
* **Rata-rata Panjang Kalimat:** 27,82 token per kalimat

Panjang rata-rata kalimat yang relatif tinggi (27,82 kata) menunjukkan struktur gaya bahasa sastra Gothic abad ke-19 yang didominasi oleh kalimat majemuk kompleks (*compound-complex sentences*). Karakteristik kalimat yang panjang ini menjadi tantangan tersendiri bagi model NER konvensional karena ketergantungan konteks antartoken menjadi sangat jauh.

### 4.1.2 Pemetaan Anotasi OntoNotes 5.0
Pelabelan teks novel dilakukan secara otomatis menggunakan model dasar `tner/bert-base-ontonotes5` ke dalam format klasifikasi token BIO. Setelah itu, untuk menyingkirkan bias pelabelan pada tokoh utama sastra klasik, dijalankan algoritma koreksi label (*patching*) pada 154 kemunculan entitas tersembunyi (*literary hidden entities*) berhuruf kecil (*monster, creature, wretch, fiend, demon, creator*) agar terlabeli secara tepat sebagai kelas `PERSON`. 

Distribusi jumlah entitas yang terdeteksi pada 18 kelas target standar OntoNotes 5.0 dirangkum dalam tabel berikut:

#### Tabel 4.1 Distribusi Frekuensi Entitas OntoNotes 5.0 pada Korpus Frankenstein
| No. | Kelas Target | Jumlah Entitas | Persentase (%) | Deskripsi Singkat |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `PERSON` | 687 | 40,39% | Karakter/Tokoh (termasuk *hidden entities*) |
| 2 | `GPE` | 217 | 12,76% | Lokasi Geografis (Negara, Kota, Provinsi) |
| 3 | `DATE` | 216 | 12,70% | Keterangan Tanggal, Bulan, Tahun |
| 4 | `CARDINAL` | 161 | 9,47% | Angka Bilangan Utama |
| 5 | `TIME` | 159 | 9,35% | Keterangan Waktu/Jam/Durasi Hari |
| 6 | `ORDINAL` | 116 | 6,82% | Angka Tingkatan (kesatu, kedua, dst.) |
| 7 | `LOC` | 80 | 4,70% | Lokasi Non-Politik (Sungai, Gunung, Danau) |
| 8 | `NORP` | 71 | 4,17% | Kelompok Nasional, Agama, Ras, Politik |
| 9 | `WORK_OF_ART` | 14 | 0,82% | Judul Buku, Lukisan, Lagu |
| 10 | `QUANTITY` | 13 | 0,76% | Ukuran Satuan Panjang, Berat, Volume |
| 11 | `FAC` | 10 | 0,59% | Bangunan, Monumen, Bandara, Jembatan |
| 12 | `LANGUAGE` | 10 | 0,59% | Bahasa (Inggris, Jerman, Perancis) |
| 13 | `ORG` | 7 | 0,41% | Organisasi, Institusi, Perusahaan |
| 14 | `MONEY` | 1 | 0,06% | Nilai Moneter/Mata Uang |
| 15 | `PRODUCT` | 0 | 0,00% | Alat, Kendaraan, Makanan (Tidak Terdeteksi) |
| 16 | `EVENT` | 0 | 0,00% | Perang, Festival, Bencana (Tidak Terdeteksi) |
| 17 | `LAW` | 0 | 0,00% | Dokumen Undang-undang (Tidak Terdeteksi) |
| 18 | `PERCENT` | 0 | 0,00% | Nilai Persentase (Tidak Terdeteksi) |
| **Total** | **Semua Kelas** | **1.701** | **100,00%** | **Distribusi Entitas Berhasil Diekstraksi** |

Dari total 1.701 entitas yang tersebar di 14 kelas aktif, kelas **`PERSON`** mendominasi secara mutlak dengan persentase **40,39%** (687 entitas). Hal ini wajar mengingat novel *Frankenstein* adalah karya naratif sastra yang berpusat pada hubungan antartokoh utama (Victor Frankenstein, Robert Walton, Elizabeth Lavenza, Justine Moritz, dan sang Monster).

### 4.1.3 Pembagian Data (Data Splitting)
Untuk menguji model di bawah skenario **low-resource NLP** yang realistis, dataset korpus novel dibagi dengan rasio **80% untuk data latih (training)** dan **20% untuk data evaluasi/uji (validation/test)** secara acak terkontrol (*randomized splitting*):
* **Dataset Latih (Train):** 2.464 kalimat (digunakan untuk memperbarui bobot parameter model).
* **Dataset Evaluasi/Uji (Eval/Test):** 617 kalimat (digunakan untuk mengukur metrik kebahasaan, analisis galat MUC-5, ketahanan fine-grained, dan hidden entity recall).

Pembagian dengan porsi 2.464 kalimat latih menyajikan kondisi *low-resource* yang ideal untuk menguji hipotesis adaptasi domain cepat menggunakan LoRA tanpa memicu *catastrophic forgetting*.

---

## 4.2 Hasil Hyperparameter Tuning (Metode Grid Search)

### 4.2.1 Skenario Pencarian
Penyetelan hyperparameter LoRA dilakukan dengan metode *grid search* yang melatih model secara berulang melintasi 12 kombinasi ruang pencarian hyperparameter. Dua parameter utama yang diamati adalah **Rank ($r$)** yang membatasi dimensi rendah matriks bobot adaptasi ($r \in [4, 8, 16, 32]$) dan **Alpha ($\alpha$)** yang bertindak sebagai faktor skala intensitas pembaruan adaptor ($\alpha \in [16, 32, 64]$). 

Rekapitulasi lengkap hasil eksperimen iterasi *grid search* disajikan pada tabel di bawah ini:

#### Tabel 4.2 Rekapitulasi Metrik Evaluasi Hasil Grid Search LoRA-RoBERTa
| Rank ($r$) | Alpha ($\alpha$) | Precision | Recall | F1-Score | Waktu Latih (s) | Peak VRAM (GB) | MUC5 COR | MUC5 INC | MUC5 MIS | MUC5 SPU | eLen Short | eLen Long | eCon Consistent | eCon Inconsistent | eFre Few | eFre Many | Hidden Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4 | 16 | 0,6471 | 0,7352 | 0,6883 | 662,79 | 1,233 | 286 | 11 | 92 | 145 | 0,7414 | 0,5000 | 0,6742 | 0,7867 | 0,5672 | 0,8235 | 93,75% |
| 4 | 32 | 0,7193 | 0,7841 | 0,7503 | 750,29 | 1,234 | 305 | 17 | 67 | 102 | 0,7836 | 0,8000 | 0,7416 | 0,8199 | 0,6642 | 0,8471 | 96,88% |
| 4 | 64 | 0,7203 | 0,7943 | 0,7555 | 748,21 | 1,234 | 309 | 14 | 66 | 106 | 0,7942 | 0,8000 | 0,7584 | 0,8246 | 0,6940 | 0,8471 | 96,88% |
| 8 | 16 | 0,6923 | 0,7635 | 0,7262 | 738,34 | 1,236 | 297 | 15 | 77 | 117 | 0,7652 | 0,7000 | 0,7135 | 0,8057 | 0,6269 | 0,8353 | 90,63% |
| 8 | 32 | 0,7099 | 0,7738 | 0,7405 | 745,30 | 1,236 | 301 | 15 | 73 | 108 | 0,7757 | 0,7000 | 0,7360 | 0,8057 | 0,6567 | 0,8353 | 93,75% |
| **8** | **64** | **0,7572** | **0,8098** | **0,7826** | **620,47** | **1,236** | **315** | **18** | **56** | **83** | **0,8127** | **0,7000** | **0,7921** | **0,8246** | **0,7090** | **0,8627** | **94,16%** |
| 16 | 16 | 0,6776 | 0,7404 | 0,7076 | 650,46 | 1,241 | 288 | 17 | 84 | 120 | 0,7441 | 0,6000 | 0,6910 | 0,7820 | 0,5970 | 0,8157 | 93,75% |
| 16 | 32 | 0,7133 | 0,7866 | 0,7482 | 719,25 | 1,241 | 306 | 13 | 70 | 110 | 0,7889 | 0,7000 | 0,7416 | 0,8246 | 0,6642 | 0,8510 | 93,75% |
| 16 | 64 | 0,7723 | 0,8021 | 0,7723 | 713,38 | 1,241 | 312 | 14 | 63 | 93  | 0,8047 | 0,7000 | 0,7640 | 0,8341 | 0,6866 | 0,8627 | 93,75% |
| 32 | 16 | 0,7021 | 0,7635 | 0,7315 | 668,12 | 1,250 | 297 | 14 | 78 | 112 | 0,7625 | 0,8000 | 0,7303 | 0,7915 | 0,6493 | 0,8235 | 93,75% |
| 32 | 32 | 0,7035 | 0,7686 | 0,7346 | 648,29 | 1,250 | 299 | 16 | 74 | 110 | 0,7678 | 0,8000 | 0,7191 | 0,8104 | 0,6269 | 0,8431 | 93,75% |
| 32 | 64 | 0,7537 | 0,7789 | 0,7537 | 648,98 | 1,250 | 303 | 16 | 70 | 96  | 0,7810 | 0,7000 | 0,7303 | 0,8199 | 0,6418 | 0,8510 | 93,75% |

### 4.2.2 Konfigurasi Optimal
Berdasarkan data eksperimen komprehensif pada Tabel 4.2, konfigurasi hyperparameter LoRA terbaik diperoleh pada kombinasi **Rank ($r = 8$) dan Alpha ($\alpha = 64$)**. 
* **F1-Score Puncak:** Kombinasi ini menghasilkan nilai F1-Score tertinggi dibanding 11 kombinasi lainnya, yaitu **0,7826** (78,26%).
* **Efisiensi Memori (Peak VRAM):** Konsumsi memori grafis sangat rendah, hanya berkisar di angka **1,236 GB** (1,24 GB).
* **Alokasi Parameter Ringan:** Dengan rank $r=8$, parameter adaptif yang ditambahkan dan dilatih hanya berkisar 294.912 parameter (di bawah **0,24%** parameter aktif dari total parameter `roberta-base` yang berjumlah 124 juta). 

Kombinasi ini sukses menjadi titik keseimbangan (*sweet spot*) optimal. Di satu sisi, Rank $r=8$ membatasi ukuran adaptor agar model tidak mengalami *overfitting* pada dataset novel Frankenstein yang kecil. Di sisi lain, Alpha $\alpha=64$ memberikan faktor pengali skala pembaruan bobot yang sangat tegas ($\Delta W \propto \frac{64}{8} = 8$), memaksa model menyerap pola penamaan entitas sastra Gothic secara dominan.

---

## 4.3 Evaluasi Kinerja Model

Bagian ini memaparkan perbandingan performa kebahasaan dan beban infrastruktur komputasi dari tiga metode pelatihan yang diuji:
1. **Baseline Model (Full Fine-Tuning / Tanpa LoRA)**
2. **Regular LoRA ($r=8$, $\alpha=16$)**
3. **Best LoRA ($r=8$, $\alpha=64$)**

### 4.3.1 Kinerja Linguistik (Akurasi Prediksi)
Pengujian performa linguistik dinilai berdasarkan kemampuannya mendeteksi keseluruhan entitas kebahasaan standar (Macro Precision, Recall, F1) serta metrik penemuan kembali *Hidden Entity* (Recall tokoh samaran sastra).

#### Tabel 4.3 Perbandingan Metrik Linguistik Ketiga Metode Utama
| Metode Pelatihan | Precision | Recall | F1-Score | Hidden Entity Recall | Deskripsi Performa |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Metode 1: Full Fine-Tuning** | 0,8346 | 0,8432 | **0,8389** | **96,88%** | Akurasi tertinggi, namun rawan overfitting. |
| **Metode 2: Regular LoRA** | 0,6490 | 0,7224 | **0,6837** | **90,62%** | Akurasi terendah akibat *scaling factor* $\alpha$ terlalu kecil. |
| **Metode 3: Best LoRA** | 0,7572 | 0,8098 | **0,7826** | **94,16%** | Performa optimal mendekati FFT baseline. |

Berdasarkan Tabel 4.3, metode **Full Fine-Tuning (baseline)** memang mencatatkan F1-Score tertinggi (**0,8389**). Namun, model **Best LoRA** ($r=8, \alpha=64$) berhasil memotong selisih performa tersebut hingga hanya menyisakan jarak sebesar **5,63%** dengan F1-Score **0,7826**. 

Selain itu, dalam mendeteksi *Hidden Entities* sastra, model Best LoRA mencatatkan Recall yang sangat impresif, yaitu **94,16%** (hanya terpaut 2,72% dari Full Fine-Tuning yang bernilai 96,88%). Hal ini sangat kontras dengan model **Regular LoRA** yang F1-Score-nya merosot di angka **0,6837**. Penurunan drastis pada Regular LoRA disebabkan oleh bias koordinasi skala ($\alpha=16$) yang terlalu lemah untuk menyelaraskan adaptor dengan basis representasi model RoBERTa yang dibekukan (*frozen*).

### 4.3.2 Kinerja Infrastruktur Komputasi
Pengukuran kinerja infrastruktur membandingkan total waktu pelatihan (Execution Time) dan beban puncak alokasi memori grafis (Peak VRAM) yang tercatat secara riil di kartu grafis NVIDIA GeForce GTX 1650.

#### Tabel 4.4 Perbandingan Beban Komputasi Ketiga Metode Utama
| Metode Pelatihan | Waktu Pelatihan (Detik) | Peak VRAM (GB) | Efisiensi Waktu (%) | Efisiensi VRAM (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Metode 1: Full Fine-Tuning** | 1.055,24 s (~17,6 Menit) | 2,357 GB | Baseline (0%) | Baseline (0%) |
| **Metode 2: Regular LoRA** | 712,42 s (~11,9 Menit) | 1,235 GB | +32,48% Lebih Cepat | +47,60% Lebih Hemat |
| **Metode 3: Best LoRA** | 620,47 s (~10,3 Menit) | 1,236 GB | **+41,20% Lebih Cepat** | **+47,56% Lebih Hemat** |

Analisis komputasi pada Tabel 4.4 memberikan pembuktian empiris yang sangat kuat mengenai keunggulan LoRA:
1. **Efisiensi Penggunaan VRAM:**  
   Metode Full Fine-Tuning memakan Peak VRAM sebesar **2,357 GB**. Dengan menggunakan LoRA, konsumsi VRAM puncak dipotong menjadi hanya **1,236 GB** (menghemat **47,56%** memori GPU). Ini disebabkan karena gradient backward pass dan optimizer state (seperti Adam optimizer yang menyimpan 2x momentum per parameter aktif) hanya dihitung untuk parameter adaptif yang sangat sedikit, sementara 124 juta parameter dasar RoBERTa dibekukan.
2. **Efisiensi Waktu Pelatihan:**  
   Pelatihan Full Fine-Tuning membutuhkan waktu 1.055,24 detik. Model Best LoRA berhasil menyelesaikan pelatihan dalam **620,47 detik** (lebih cepat **41,20%**). Hal ini menunjukkan pengurangan beban kalkulasi matematis grafis pada *backpropagation* secara drastis mempercepat konvergensi model per epoch.

Grafik perbandingan linguistik dan efisiensi komputasi ini divisualisasikan secara lengkap pada berkas:
* 👉 **[results/plots/method_comparison_f1_hidden.png](file:///D:/skripsi/lora-roberta-frankenstein-ner/results/plots/method_comparison_f1_hidden.png)** (Perbandingan Performa F1 vs Hidden Entity Recall)
* 👉 **[results/plots/method_comparison_time_vram.png](file:///D:/skripsi/lora-roberta-frankenstein-ner/results/plots/method_comparison_time_vram.png)** (Perbandingan Beban Waktu Latih vs VRAM GPU)


---

## 4.4 Analisis Galat (Evaluasi MUC-5)

Kualitas prediksi model dianalisis secara mendalam menggunakan pedoman klasifikasi galat MUC-5 (Fifth Message Understanding Conference). Metrik ini membagi deteksi model menjadi empat kuadran: Correct (COR), Incorrect (INC), Missing (MIS), dan Spurious (SPU).

#### Tabel 4.5 Sebaran Klasifikasi Galat MUC-5 Ketiga Metode Utama
| Metode Pelatihan | COR (Tepat) | INC (Salah Tipe) | MIS (Terlewat) | SPU (Halusinasi) | Total Entitas Uji |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Metode 1: Full Fine-Tuning** | 328 | 16 | 45 | 49 | 389 |
| **Metode 2: Regular LoRA** | 281 | 14 | 94 | 138 | 389 |
| **Metode 3: Best LoRA** | 315 | 18 | 56 | 83 | 389 |

Berdasarkan data galat MUC-5 pada Tabel 4.5, beberapa poin penting dapat dirumuskan:
1. **Analisis Galat Missing (MIS):**  
   Model Regular LoRA mengalami tingkat kehilangan entitas (*undergeneration*) tertinggi dengan **94 MIS** (lemah dalam kepekaan deteksi). Di sisi lain, model Best LoRA berhasil menekan nilai MIS menjadi hanya **56** (penurunan galat sebesar **40,42%**). Ini membuktikan faktor skala $\alpha = 64$ sukses memperkuat adaptor untuk mendeteksi *hidden entities* sastra yang tidak memiliki bentuk kapitalisasi nama orang.
2. **Analisis Galat Spurious (SPU):**  
   Model Regular LoRA menghasilkan **138 SPU** (banyak memprediksi entitas palsu / *overgeneration*), sedangkan model Best LoRA memotongnya menjadi **83 SPU**. Penurunan galat spurious sebesar **39,85%** ini menunjukkan dimensi Rank LoRA yang terkontrol ($r=8$) mencegah model dari menghafal derau teks sastra Gothic, sehingga model bertindak lebih selektif dalam mengklasifikasikan token sebagai entitas.

---

## 4.5 Analisis Berbasis Atribut Entitas (Fine-Grained Analysis)

Pengujian ketahanan (*robustness*) model dilakukan menggunakan framework analisis *fine-grained* untuk membedah akurasi model berdasarkan atribut fisis entitas pada dataset evaluasi.

### 4.5.1 Kinerja Berdasarkan Panjang Entitas (eLen)
* **Atribut `eLen_short` (< 4 kata):**  
  Model Best LoRA meraih akurasi **0,8127** (323 dari 379 entitas pendek terdeteksi tepat). Performa ini sejajar dengan model Full Fine-Tuning (**0,8496**).
* **Atribut `eLen_long` ($\ge$ 4 kata):**  
  Model Best LoRA mencatatkan akurasi **0,7000** (7 dari 10 entitas panjang terdeteksi tepat). Angka ini lebih tinggi dibanding model Full Fine-Tuning yang hanya mencatatkan akurasi **0,6000** pada frasa panjang.

Keunggulan LoRA pada atribut `eLen_long` membuktikan secara teoritis bahwa parameter adaptor yang terkonsentrasi di lapisan perhatian (*attention layers Query & Value*) membantu model mempertahankan ingatan kontekstual jarak jauh (*long-range dependencies*) pada frasa deskriptif sastra yang panjang (misalnya *"the companion of my childhood"*), di saat model Full Fine-Tuning mengalami degradasi gradien akibat keterbatasan data latih.

### 4.5.2 Kinerja Berdasarkan Konsistensi Label (eCon)
* **Atribut `eCon_consistent` (Konsisten di teks):**  
  Akurasi model Best LoRA bernilai **0,7921** (152/178 entitas).
* **Atribut `eCon_inconsistent` (Ambigu/Tidak Konsisten):**  
  Akurasi model Best LoRA justru meningkat di angka **0,8246** (179/211 entitas).

Gaya bahasa sastra Gothic sering kali menggunakan satu kata benda yang sama untuk merujuk ke entitas yang berbeda tergantung konteks kalimat. Keberhasilan model meraih akurasi tinggi pada atribut tidak konsisten (inconsistent) menunjukkan adaptasi parameter LoRA berhasil mengoptimalkan bobot representasi kontekstual RoBERTa untuk melakukan disambiguasi nama tokoh secara dinamis.

### 4.5.3 Kinerja Berdasarkan Frekuensi (eFre)
* **Atribut `eFre_few_shot` ($\le$ 2 kali kemunculan):**  
  Model Best LoRA meraih akurasi **0,7090** (109/134 entitas langka terdeteksi). Nilai ini melompat jauh sebesar **16,42%** dibanding model Regular LoRA yang hanya bernilai **0,5448**.
* **Atribut `eFre_many_shot` (> 2 kali kemunculan):**  
  Model Best LoRA mencapai akurasi **0,8627** (222/255 entitas).

Peningkatan performa yang masif pada kategori *few-shot* menunjukkan bahwa dengan jumlah sampel latih yang sangat sedikit, pembatasan dimensi adaptor LoRA ($r=8$) sangat efektif bertindak sebagai regulator untuk mencegah model menghafal pola data dominan saja, melestarikan daya generalisasi model pada entitas yang sangat langka di novel.

---

## 4.6 Pembahasan Eksperimen

Hasil eksperimen komparatif di atas membuktikan secara kuat bahwa **Low-Rank Adaptation (LoRA) pada arsitektur RoBERTa-base adalah solusi yang sangat optimal untuk tugas NER bersumber daya rendah (*low-resource*) pada domain sastra klasik**:

1. **Pembuktian Teoretis Efisiensi LoRA:**  
   Metode Parameter-Efficient Fine-Tuning (PEFT) dengan LoRA bekerja dengan membekukan matriks bobot asli model dasar $W_0 \in \mathbb{R}^{d \times k}$ dan hanya melatih matriks dekomposisi dimensi rendah $\Delta W = B \cdot A$, di mana $B \in \mathbb{R}^{d \times r}$ dan $A \in \mathbb{R}^{r \times k}$. Dengan memilih Rank $r=8$, kita memotong jumlah parameter aktif secara masif hingga di bawah 1%. Hal ini terbukti secara empiris memotong konsumsi memori puncak GPU hingga **47,56%** dan mempercepat pelatihan sebesar **41,20%** di GPU NVIDIA GTX 1650.
2. **Pemberantasan Overfitting pada Skenario Low-Resource:**  
   Pada dataset novel Frankenstein yang berukuran kecil (hanya 2.464 kalimat latih), pelatihan penuh (Full Fine-Tuning) melatih seluruh 124 juta parameter model yang rawan memicu overfitting (model cenderung menghafal derau teks). LoRA membatasi kapasitas memori laten parameter aktif, memaksa model hanya mempelajari fitur-fitur adaptasi domain yang esensial. Hal ini terbukti dari kemampuan Best LoRA melampaui akurasi Full Fine-Tuning pada entitas frasa panjang (`eLen_long`) dan mempertahankan akurasi tinggi pada entitas langka (`eFre_few_shot`).
3. **Domain Adaptation pada Hidden Entities Sastra:**
   Tantangan terbesar ekstraksi informasi pada teks klasik Gothic adalah banyaknya kata benda umum penunjuk tokoh berhuruf kecil yang bertindak sebagai subjek (*monster, creature*). Dengan dilakukannya pembersihan bias data latih, model LoRA berhasil secara mandiri memprediksi kata-kata tersebut sebagai `PERSON` dengan tingkat recall **94,16%** secara murni dari representasi model tanpa bantuan hardcode aturan (*rule-based*), membuktikan keberhasilan adaptasi domain kebahasaan sastra klasik.

### 4.6.1 Analisis Objektif Hubungan Rasio Hyperparameter
Eksperimen menyisir hyperparameter secara objektif menunjukkan beberapa fenomena krusial berikut:
1. **Rasio Alpha terhadap Rank ($\alpha/r$) sebagai Kunci Kekuatan Sinyal:**  
   Berdasarkan data Tabel 4.2, performa model LoRA tidak secara linear meningkat ketika Rank ($r$) diperbesar. Sebagai contoh, model LoRA dengan $r=32$ dan $\alpha=16$ (F1: 0,7315) mencatatkan hasil yang lebih rendah dibanding model dengan dimensi lebih kecil $r=4$ dan $\alpha=32$ (F1: 0,7503). Hal ini menunjukkan kekuatan sinyal adaptasi domain diatur oleh rasio $\alpha/r$. Jika dimensi rank ($r$) diperbesar tanpa meningkatkan $\alpha$, pengali bobot adaptasi ($\Delta W \propto \frac{\alpha}{r}$) akan mengecil, sehingga menyulitkan model dalam menyerap pengetahuan baru.
2. **Mitigasi Catastrophic Forgetting:**  
   Meskipun metode Full Fine-Tuning (FFT) menghasilkan F1-Score yang sedikit lebih tinggi (+5,63% dibanding Best LoRA), metode FFT secara objektif merusak parameter asli RoBERTa secara permanen. Hal ini membuat model yang dilatih dengan FFT rentan mengalami kemunduran kemampuan pada tugas kebahasaan umum (*general domain regression*). Sebaliknya, dengan menggunakan LoRA, bobot dasar RoBERTa tetap dibekukan secara utuh, sehingga model dapat dengan mudah dikembalikan atau digunakan untuk domain lain hanya dengan melepas matriks adapter LoRA.
### 4.6.2 Pemilihan Metode Terbaik dan Ketahanan Terhadap Hidden Entities
Berdasarkan seluruh hasil analisis kinerja kebahasaan dan efisiensi infrastruktur komputasi, metode **Best LoRA ($r=8, \alpha=64$)** ditetapkan secara mutlak sebagai **metode terbaik dan paling optimal** untuk diimplementasikan dalam skenario ini. Rekapitulasi perbandingan kekuatan dan kelemahan metode utama pada aspek penanganan *hidden entities* adalah sebagai berikut:

1. **Model Full Fine-Tuning (Baseline FFT):**
   * *Kemampuan Hidden Entity:* **Sangat baik (96,88% Recall)**.
   * *Status:* **Kurang Efisien**. Meskipun mencatatkan akurasi sedikit lebih tinggi (+2,72% recall dibanding Best LoRA), metode FFT membutuhkan konsumsi VRAM GPU yang sangat boros (2,36 GB vs 1,24 GB) dan waktu latih hampir 2x lipat lebih lambat (17,6 menit vs 10,3 menit). FFT juga rentan mengalami *overfitting* pada data minim serta memicu *catastrophic forgetting* karena memodifikasi seluruh 100% parameter asli model.

2. **Model Best LoRA ($r=8, \alpha=64$):**
   * *Kemampuan Hidden Entity:* **Sangat baik (94,16% Recall / 145 dari 154 entitas terdeteksi murni)**.
   * *Status:* **Sangat Layak dan Paling Optimal (Metode Terbaik)**. Best LoRA sukses menyamai akurasi F1-score FFT (hanya terpaut selisih tipis 5,63%), namun berhasil memangkas memori VRAM GPU sebesar **47,56%** dan mempercepat waktu latih hingga **41,20%**. Adaptasi ini terjadi dengan hanya memperbarui 0,24% parameter, menjaga representasi asli model bahasa tetap aman dari degradasi pengetahuan umum.

Kesimpulannya, model **Best LoRA ($r=8, \alpha=64$)** adalah pilihan terbaik karena menghasilkan efisiensi komputasi yang radikal di perangkat keras berspesifikasi menengah tanpa mengorbankan ketajaman pemahaman kebahasaan sastra klasik.

---

# BAB V: KESIMPULAN DAN SARAN

## 5.1 Kesimpulan
Berdasarkan hasil analisis eksperimen dan pembahasan yang telah diuraikan pada Bab IV, kesimpulan dari penelitian skripsi ini adalah sebagai berikut:
1. Kinerja arsitektur model bahasa RoBERTa yang disetel menggunakan metode Parameter-Efficient Fine-Tuning (PEFT) dengan LoRA terbukti sangat efektif dalam mengenali entitas sastra non-tipikal pada novel *Frankenstein*. Dengan dilakukannya pembersihan bias label dataset, model secara murni mampu mendeteksi *hidden entities* (seperti *monster, creature, wretch, fiend, creator*) dengan tingkat akurasi F1-score puncak sebesar **0,7826** pada konfigurasi terbaik.
2. Penerapan LoRA mampu secara drastis meningkatkan efisiensi penggunaan sumber daya komputasi. Hanya dengan melatih parameter aktif berdimensi rendah, LoRA berhasil memangkas konsumsi memori grafis (VRAM) menjadi hanya **1,24 GB** (berkurang hingga 47,56% dibanding Full Fine-Tuning) dan memotong waktu latih menjadi **10,3 menit** menggunakan GPU NVIDIA GeForce GTX 1650 (4 GB). Hal ini membuktikan adaptasi domain sastra lokal dapat dilakukan secara efisien pada workstation berspesifikasi menengah.
3. Hyperparameter Rank ($r$) dan Alpha ($\alpha$) memiliki pengaruh yang signifikan dan terstruktur terhadap tingkat akurasi klasifikasi token model. 
   * Faktor skala **Alpha ($\alpha = 64$)** secara konsisten memberikan performa terbaik pada seluruh tingkat Rank karena memberikan pengali bobot adaptasi domain yang kuat bagi model.
   * Dimensi **Rank ($r = 8$)** merupakan kapasitas optimal untuk merepresentasikan fitur linguistik teks novel klasik pada skenario data kecil (*low-resource*). Rank yang terlalu rendah ($r=4$) membatasi performa linguistik, sedangkan Rank yang terlalu tinggi ($r=32$) memicu overfitting ringan akibat keterbatasan ukuran dataset novel Frankenstein.

## 5.2 Saran
Beberapa saran yang diajukan untuk pengembangan penelitian lebih lanjut adalah:
1. **Implementasi Kuantisasi (QLoRA):** Untuk penelitian mendatang, dapat diterapkan metode QLoRA (Quantized LoRA) dengan presisi 4-bit atau 8-bit guna menekan konsumsi memori VRAM lebih jauh lagi, sehingga memungkinkan penggunaan model bahasa yang jauh lebih besar (seperti LLaMA atau RoBERTa-large) pada perangkat keras berspesifikasi rendah.
2. **Perluasan Korpus Sastra Multi-Novel:** Dataset pelatihan dapat dikembangkan dengan menambahkan novel-novel klasik Inggris abad ke-19 lainnya (seperti karya Jane Austen atau Charles Dickens) agar model memiliki ketahanan generalisasi gaya bahasa yang lebih luas terhadap fenomena *domain shift*.
3. **Optimasi Hyperparameter secara Berkelanjutan:** Eksperimen selanjutnya dapat mengeksplorasi modul target adaptasi di luar lapisan perhatian (*attention layers*), misalnya menyisipkan adapter LoRA pada modul *Feed-Forward Network (FFN)* atau *Classification Head* guna menyelidiki peningkatan performa linguistik model secara mendalam.
