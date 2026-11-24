# Kontribusi

**Mata Kuliah:** Deep Learning  
**Tugas:** Ujian Tengah Semester (UTS) - Group Task (Kelompok 7)  
**Tanggal Pengumpulan:** 24 November 2026  

## Peran anggota

| Anggota | Peran | Pull request |
|---|---|---|
| Abdu Fattah Hidayatullah (`Elheims`, 101032300039) | **Task 2 (RUL regression):** EDA & preprocessing, pelabelan, sliding windows, baseline, 1D-CNN, GRU, validasi & 5-fold CV, uncertainty & aturan servis, model final (`prediksi_rul.csv`), bagian laporan Task 2, skrip demo lisan (`demo.py`), serta integrasi repositori dan laporan akhir | [#1](https://github.com/Elheims/Midterm_DL_Kel7/pull/1), [#2](https://github.com/Elheims/Midterm_DL_Kel7/pull/2), [#3](https://github.com/Elheims/Midterm_DL_Kel7/pull/3), [#4](https://github.com/Elheims/Midterm_DL_Kel7/pull/4), [#5](https://github.com/Elheims/Midterm_DL_Kel7/pull/5), [#6](https://github.com/Elheims/Midterm_DL_Kel7/pull/6), [#8](https://github.com/Elheims/Midterm_DL_Kel7/pull/8), [#10](https://github.com/Elheims/Midterm_DL_Kel7/pull/10), [#11](https://github.com/Elheims/Midterm_DL_Kel7/pull/11), [#12](https://github.com/Elheims/Midterm_DL_Kel7/pull/12) |
| Muhammad Aidil Fadhli Awalludin (`aidilfadhli`, 101032300087) | **Task 1 (Multimodal Classification):** EDA & visualisasi multimodal (citra + sinyal sensor), desain arsitektur CNN 2D, 1D-CNN, Late Fusion & Cross-Attention, eksperimen ketahanan (label smoothing, mitigasi domain shift per lini, missing sensor / modality dropout), pengujian 3 random seeds, pembuatan `prediksi_klasifikasi.csv`, dan bagian laporan Task 1 | [#7](https://github.com/Elheims/Midterm_DL_Kel7/pull/7), [#9](https://github.com/Elheims/Midterm_DL_Kel7/pull/9) |
| Sandy Hidayatullah (`sandyxd18`, 101032300009) | **Task 3 (Multimodal Clustering):** eksplorasi distorsi mikroskop & spektrum, unsupervised representation learning, fusi dua modalitas, deep clustering & penentuan jumlah cluster, visualisasi UMAP/t-SNE, pembuatan `cluster.csv`, dan bagian laporan Task 3 | [#13](https://github.com/Elheims/Midterm_DL_Kel7/pull/13) |

> Catatan workflow GitHub: repositori kelompok berada di akun `Elheims`, dan GitHub tidak mengizinkan pemilik repositori mem-fork repositorinya sendiri. Karena itu `Elheims` bekerja melalui branch dan membuka pull request ke `main`, sedangkan anggota lain (`aidilfadhli` dan `sandyxd18`) bekerja melalui fork masing-masing lalu membuka pull request ke `main`.

## Penggunaan AI tools

> Penggunaan AI harus disetujui seluruh anggota kelompok. Setiap penggunaan mencantumkan: tool, bagian yang dibantu, dan cara verifikasi.

| Tool | Bagian yang dibantu | Cara verifikasi |
|---|---|---|
| Antigravity / Gemini (Google) | **Task 1 (Multimodal Classification):** pembuatan kode pipeline (`dataset.py`, `models.py`, `train_eval.py`, `run_experiments.py`, `generate_eda_figures.py`), perancangan blok Residual 1D/2D dan strategi fusi (*Late Fusion* & *Cross-Attention*), pembuatan visualisasi EDA & perbandingan modalitas, perbaikan bug *float16 overflow* pada statistik sensor, serta penyusunan draf bagian laporan Task 1 dan lembar kontribusi. | Menjalankan *sanity check* (*overfitting* 1 batch 32 sampel hingga loss $< 0{,}05$ dan Macro-F1 $= 1{,}00$), memeriksa dimensi tensor & nilai *finite* statistik `float64` tanpa *overflow*, memverifikasi metrik Macro-F1 dengan `scikit-learn`, mengecek distribusi prediksi seluruh 7 kelas pada `prediksi_klasifikasi.csv`, serta *code review* mandiri dan silang antar anggota kelompok sebelum *merge*. |
| Claude Code (Anthropic) | **Task 2 (RUL):** penulisan kode pipeline (`eda.py`, `data.py`, `labels.py`, `windows.py`, `evaluate.py`, `baseline.py`, `models.py`, `train_nn.py`, `deep.py`, `error_analysis.py`, `validation.py`, `uncertainty.py`, `maintenance.py`, `final.py`, `predict.py`, `demo.py`), pembuatan grafik, dan penyusunan draf bagian laporan Task 2 (`task2_rul.tex`, `lampiran_task2.tex`). | Menjalankan uji kausalitas & bebas NaN (`check_preprocess.py`), uji label (`labels.py`), uji kebocoran jendela termasuk kasus bocor buatan (`windows.py`), uji overfit satu batch pada semua model deep learning, reproduksi penuh `python run_all.py --task 2` (dua kali run menghasilkan angka & `prediksi_rul.csv` identik), pengecekan angka laporan terhadap JSON via `python -m task2_rul.demo`, serta pemeriksaan visual semua grafik dan log keputusan di `catatan/keputusan.md`. |
| Antigravity / Gemini (Google) | **Review Task 2:** audit kesesuaian Task 2 terhadap rubrik UTS, pemeriksaan format `prediksi_rul.csv`, pembaruan `kontribusi.md` (bagian Task 2), dan penyusunan panduan penjelasan presentasi lisan (`catatan/panduan_presentasi_task2.md`). | Seluruh temuan dan angka diverifikasi langsung terhadap file kode, file JSON hasil di `kode/results/task2/`, dan `prediksi_rul.csv`. |
| Claude Code (Anthropic, model Claude Sonnet 5.5) | **Task 3 (Multimodal Clustering):** pembacaan spesifikasi dan dataset, penulisan seluruh kode notebook `kode/task3_clustering/task3_clustering.ipynb` (pra-pemrosesan, augmentasi, SimCLR citra dan spektrum, fusi `concat`/`joint`, KMeans, GMM, Agglomerative, pembelahan graf kNN, DEC, kriteria pemilihan K, diagnostik, UMAP/t-SNE), skrip `extract_outputs.py`, perbaikan bug (broadcasting augmentasi spektrum, detektor spike, overflow fp16 pada loss, presisi campuran untuk GPU T4), pembacaan output run dan gambar, serta draf catatan Task 3 pada berkas ini. | Run penuh dijalankan sendiri oleh Sandy di Google Colab (GPU T4), dan angka yang dilaporkan diambil dari output notebook itu. Smoke test lokal dengan `FAST = True`. Karena tidak ada label, kebenaran cluster tidak bisa diperiksa langsung, jadi hanya diagnostik internal yang dipakai (silhouette, Davies-Bouldin, Calinski-Harabasz, BIC, stabilitas bootstrap, ARI lintas modalitas, `knn_cut`, d' per modalitas, kebocoran mikroskop). K dipaksa 7 (`K_OVERRIDE`) walau kriteria memilih 5, dan aturan pemilihan metode akhir dibuat setelah melihat hasil. Keduanya dicatat apa adanya di PR dan laporan. |

## Persetujuan anggota

Seluruh anggota kelompok menyatakan bahwa pekerjaan yang dikumpulkan adalah hasil kerja orisinal kelompok, menyetujui pembagian peran dan catatan penggunaan AI di atas, serta mampu menjelaskan seluruh isi laporan dan kode pada sesi ujian lisan (*oral exam*).

Centang sebelum merge final ke `main`:

- [x] Elheims
- [x] aidilfadhli
- [ ] sandyxd18
