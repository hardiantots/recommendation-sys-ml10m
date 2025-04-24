# Laporan Proyek Machine Learning - Hardianto Tandi Seno

## Project Overview
Layanan streaming film seperti Netflix, Disney+, dan Amazon Prime Video telah menjadi bagian yang tak terpisahkan dari kehidupan sehari-hari. Dengan semakin luasnya pilihan film dan serial yang tersedia, tantangan utama yang dihadapi oleh platform-platform ini ataupun platform sejenisnya yaitu bagaimana cara menyajikan konten yang relevan dan menarik bagi setiap pengguna secara personal. Dalam konteks ini, sistem rekomendasi memainkan peranan penting sebagai alat yang mampu meningkatkan pengalaman pengguna, memperpanjang waktu tonton, serta meningkatkan loyalitas terhadap platform.
Secara garis besar, sistem rekomendasi bekerja dengan membantu pengguna menemukan konten yang relevan di antara banyaknya pilihan konten yang tersedia berdasarkan riwayat tontonan ataupun perilaku user lain yang serupa. Content-based filtering dan collaborative filtering merupakan dua pendekatan utama yang sering digunakan dalam pengembangan sistem rekomendasi ini.

Content-based filtering lebih menganalisis fitur dari item (dalam hal ini, film) yang telah disukai oleh pengguna, lalu merekomendasikan film lain dengan karakteristik serupa. Sementara itu, collaborative filtering mengandalkan pola perilaku pengguna secara kolektif, dengan mengasumsikan bahwa pengguna yang memiliki preferensi serupa di masa lalu kemungkinan besar akan menyukai konten yang sama di masa depan.

Salah satu dataset yang paling populer dan banyak digunakan untuk mengembangkan dan menguji sistem rekomendasi berbasis pendekatan ini adalah MovieLens, yang disediakan oleh GroupLens Research. Dataset ini berisi informasi tentang nilai film yang diberikan oleh sejumlah pengguna, serta metadata seperti genre, judul, dan tahun rilis film. Dengan menggunakan dataset ini, peneliti dan praktisi dapat membangun model rekomendasi, melakukan evaluasi performa sistem, dan mengembangkan serta menguji sistem rekomendasi berdasarkan 2 pendekatan tersebut.

Penelitian yang dilakukan oleh Department of Computer Science, California State Polytechnic University, Pomona, CA, USA et al. pada tahun 2021 dalam mengimplementasikan teknik Collaborative Filtering untuk membuat sistem rekomendasi film dengan menggunakan dataset dari MovieLens memberikan hasil yang baik ketika teknik tersebut diterapkan karena dapat meningkatkan akurasi rekomendasi [1]. Selain itu, terdapat juga penelitian yang dilakukan oleh Rakesh (2023) untuk membuat sistem rekomendasi film menggunakan dataset TMDB 5000. Hasilnya menyatakan bahwa secara keseluruhan, penerapan content-based filtering menunjukkan potensi pendekatan berbasis atribut untuk memberikan rekomendasi yang efisien dan personal terhadap user [2].

Dengan hasil yang telah dipaparkan dalam kedua artikel tersebut, tentunya proyek ini diharapkan dapat memberikan pemaparan untuk masing-masing teknik yang optimal digunakan untuk diterapkan pada dataset MovieLens ini ataupun dataset film yang lainnya.

## Business Understanding

Dalam industri layanan streaming film, perusahaan harus dapat memberikan pengalaman menonton yang relevan dan unik untuk pelanggan mereka karena persaingan yang ketat dan tingginya ekspektasi pengguna. Salah satu cara utama untuk mencapai hal ini adalah dengan menerapkan sistem rekomendasi yang kuat. Dengan menggunakan dataset MovieLens, perusahaan dapat mengevaluasi dan mengembangkan model rekomendasi dengan mencoba menerapkan 2 pendekatan sekaligus, yaitu content-based filtering dan collaborative filtering. Diharapkan model ini dapat lebih akurat memprediksi preferensi pengguna dan menyarankan film yang sesuai dengan selera mereka. Peningkatan jumlah konten yang dikonsumsi per pengguna, peningkatan kepuasan pelanggan, dan pengoptimalan katalog film adalah dampak bisnis yang diharapkan dari penerapan sistem ini. Dalam jangka panjang, sistem rekomendasi yang baik tidak hanya akan meningkatkan loyalitas pelanggan tetapi juga dapat meningkatkan pendapatan perusahaan melalui penurunan tingkat churn dan peningkatan langganan yang berkelanjutan.

### Problem Statements
- Bagaimana sistem rekomendasi dapat meningkatkan personalisasi pengalaman menonton pengguna pada platform layanan streaming film?
- Bagaimana perbandingan efektivitas content-based filtering dan collaborative filtering dalam kasus tertentu ketika digunakan untuk memprediksi preferensi pengguna secara akurat?

### Goals
- Membantu meningkatkan personalisasi pengalaman menonton pengguna pada platform layanan streaming film dengan sistem rekomendasi yang akan memberikan preferensi berdasarkan riwayat tontonan ataupun riwayat pengguna lainnya.
- Mengembangkan dan membandingkan efektivitas antara kedua teknik yang ada ketika digunakan dalam memprediksi preferensi pengguna dalam kasus-kasus tertentu

### Solution Statements
- Dalam mencapai tujuan ini, akan digunakan diterapkan 2 teknik (Content-Based Filtering & Collaborative Filtering) untuk membandingkan efektivitas masing-masing teknik dalam kasus tertentu ketika memprediksi preferensi pengguna.
- Dilakukan proses Normalisasi untuk kolom tertentu dan TF-IDF untuk menghitung frekuensi kemunculan kata dalam suatu kolom.
- Pengukuran performa setiap teknik akan menggunakan Cosine Similarity dan Diversity (Keberagaman)-Novelty (Kebaruan) saat item rekomendasi muncul.  

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah data film MovieLens dengan jumlah rating dalam dataset tersebut sebanyak 10M (10.000.000) yang diaplikasikan pada 10.000 film oleh sekitar 72.000 pengguna platform MovieLens untuk memberikan rating. Dataset ini diperoleh dari sumber Grouplens (https://grouplens.org/datasets/movielens/10m/). Dataset ini terdiri dari 3 file utama yang bisa digunakan untuk membangun sistem rekomendasi:

### 1. Movies Dataset (movies.dat)
* Kondisi Data:
  - Dataset terdiri dari 10.681 data dan 3 kolom
* Fitur Dataset:
  - movieId: ID unik untuk setiap film (integer)
  - title: Judul film (string) 
  - genres: Genre film, bisa berisi banyak genre (string)

### 2. Ratings Dataset (ratings.dat) 
* Kondisi Data:
  - Terdiri dari 10.000.054 data dan 4 kolom
* Fitur Dataset:
  - userId: ID unik setiap pengguna (integer)
  - movieId: ID unik setiap film yang memiliki rating (integer) 
  - rating: Rating yang diberikan pengguna kepada suatu film dengan skala 0.0-5.0 (float)
  - timestamp

### 3. Tags Dataset (tags.dat)
* Kondisi Data:
  - Terdiri dari 95.580 data dan 4 kolom
* Fitur Dataset:
  - userId: ID unik setiap pengguna (integer)
  - movieId: ID unik setiap film yang memiliki rating (integer) 
  - tag: Teks yang menggambarkan kesan pengguna terhadap film tersebut (string)
  - timestamp: 

### Tahapan Pemahaman pada Data
- Diawali dengan menggabungkan 2 dataset sebagai 1 dataset untuk melakukan analisis lebih lanjut (Movies & Ratings Dataset). Untuk Tags Dataset tidak digabung karena informasinya untuk keadaan terkini belum diperlukan
- Selanjutnya yaitu menggali infomasi pada dataset (jumlah baris-kolom, tipe data setiap kolom identifikasi missing value, melihat visualisasi distribusi, dll.). Dalam pemahaman ini dilakukan beberapa perubahan, seperti mengubah tipe data kolom timestamp (dari unix menjadi datetime), mengekstrak tahun rilis film dari judul film, menghapus film yang tidak memiliki genre, dan lain sebagainya.
- Beberapa Explorartory Data Analysis juga dilakukan dalam rangka mencoba memahami dataset yang ada, seperti melihat distribusi rating pengguna, melihat urutan Top 10 pada data (Genre Film Terbanyak, Tahun & Bulan dengan Pemberian Rating Terbanyak, Film dengan Pemberian Rating & Rata-rata Rating Terbanyak).

## Data Preparation

Singkatnya, dataset film dari MovieLens ini di-donwload dari sumber terbuka (Grouplens), kemudian dataset yang telah di-download selanjutnya di-load dengan menggunakan pandas. Alur berikutnya yaitu melakukan Data Understanding (melibatkan penggabungan beberapa dataset, pengecekan missing value, informasi bentuk data, melakukan pengubahan tipe data pada kolom tertentu, membuat variable baru untuk memahami genre film tanpa data duplikat, mengekstrak tahun rilis film dari judul, dll) dan EDA.

Setelah kedua hal tersebut selesai dilakukan, saatnya masuk dalam tahapan Data Preparation yang terbagi menjadi 3 bagian:
1. Feature Engineering (menambahkan fitur pada kolom data), dimana dalam bagian ini dihitung nilai rata-rata rating dari setiap judul film, lalu kemudian hasilnya akan di-map ke masing-masing title pada dataset hasil penggabungan. Ini dilakukan sebagai salah satu fitur untuk Content-Based Filtering. Setelah itu, variable baru dibuat untuk menampung dataset yang telah tidak memiliki duplikat (akan digunakan pada kedua teknik).

2. Persiapan data Content-Based Filtering
    - Melakukan perhitungan TF-IDF pada kolom genres yang telah melalui proses penghilangan tanda baca. Proses ini dilakukan untuk menghitung frekuensi kemunculan suatu kata dalam kumpulan data sehingga semakin tinggi nilainya, frekuensi kemunculan suatu kata akan lebih sering.
    - Melakukan normalisasi pada kolom tahun film dan rata-rata rating film untuk mengubah skala data menjadi hanya dikisaran 0 dan 1.
    - Menggabungan hasil TF-IDF dan normalisasi menjadi kolom fitur untuk Content-Based Filtering. Ini dilakukan untuk memastikan pemilihan fitur yang sesuai dengan tujuan.

3. Persiapan data Collaborative Filtering
    - Mengambil kolom data yang diperlukan sebagai fitur pada dataset yang sama untuk Content-Based Filtering.
    - Membuat pemetaan pada kolom userId dan movieId ke bentuk angka karena model machine learning (terutama berbasis matriks seperti collaborative filtering) hanya bekerja dengan angka, bukan string atau ID acak.
    - Mengganti userId dan movieId menjadi angka berdasarkan kode yang sudah dibuat agar siap dipakai untuk pembuatan matriks interaksi (ratings matrix) yang efisien dan sesuai dengan TensorFlow.
    - Menampung nilai unik dari jumlah userId dan movieId karena matriks rating akan punya ukuran [num_user x num_movie], jadi info ini penting untuk membuat matriks.
    - Membuat fungsi split dataset agar bisa melatih model di data training dan menguji performanya di data test.
    - Membuat fungsi rating sparse tensor untuk mengubah data rating jadi bentuk sparse tensor, yaitu matriks besar dengan banyak nilai nol, hanya menyimpan nilai yang ada.
    - Mengaplikasikan kedua fungsi ke dataset
    - Dilakukan juga perhitungan global_mean untuk digunakan sebagai nilai awal atau fallback prediction saat tidak ada data rating user tertentu (cold start).

## Modelling

### Model 1: Content-Based Filtering
Content-Based Filtering memiliki beberapa kelebihan, seperti rekomendasi didasarkan pada preferensi user sendiri dan kemiripan item yang pernah dia suka, bisa bekerja meski hanya ada 1 user aktif (tidak tergantung pada user-user lain), dan item baru bisa langsung direkomendasikan asalkan punya fitur deskriptif (genre, sinopsis, dll). Namun, terdapat juga beberapa kekurangan dari teknik ini seperti rekomendasi hanya akan mirip dengan apa yang pernah dilihat user (kurang eksploratif) dan kualitas rekomendasi sangat tergantung pada fitur konten yang tersedia.

Dalam proyek ini, Content-Based Filtering dilakukan dengan menggunakan model K-NN (Nearest Neigbors). Inputnya berupa fitur yang telah melalui proses normalisasi dengan fungsi MinMaxScaler() dari scikit-learn untuk mengubah skala pada fitur data tertentu menjadi lebih sempit dan fungsi TF-IDF(stop_words='english') untuk membuat pembobotan nilai terhadap suatu kata dengan menghilangkan juga stop words dalam bahasa Inggris. Model k-NN yang digunakan diakses dengan sklearn.neighbors, yang digunakan untuk menemukan item (misalnya film) yang mirip berdasarkan fitur kontennya. Beberapa parameter yang digunakan pemodelan ini yaitu metric='cosine' (bearti model akan mengukur kemiripan antar item berdasarkan cosine similarity. Cosine similarity cocok untuk data vektor seperti fitur konten film karena menghitung sudut antar vektor, bukan jaraknya secara langsung), n_neighbors=7 (model akan mencari 7 item yang paling mirip (tetangga terdekat) untuk setiap item yang ditanyakan), dan n_jobs=-1 (menggunakan semua core CPU yang tersedia agar proses pencarian tetangga lebih cepat.)

### Model 2: Collaborative Filtering
Collaborative Filtering memiliki beberapa kelebihan, seperti tidak perlu fitur konten seperti genre atau deskripsi (hanya butuh data rating user), bisa menangkap pola tersembunyi, dan B=bisa diskalakan ke dataset besar dengan pendekatan efisien seperti SGD atau Alternating Least Squares. Beberapa kelemahan yang dimiliki oleh teknik ini yaitu berkaitan dengan cold-start problem (jika belum ada data masih sulit untuk melakukan rekomendasi), matrix rating seringkali sangat kosong (~95% nol) yang bisa mempersulit pelatihan jika terlalu sedikit interaksi, dan kalau embedding terlalu besar atau data terlalu sedikit, bisa mempelajari noise.

Collaborative Filtering menggunakan input berupa fitur yang telah melalui proses train-test split dan pengubahan bentuk data agar bisa sesuai dengan rancangan model dengan menggunakan TensorFlow. Beberapa parameter utama yang digunakan berdasarkan class yang telah dibuat untuk bisa menjalankan teknik ini yaitu:
- num_users: Jumlah total user unik dalam dataset. Ini akan menentukan ukuran embedding matrix U.
- num_movies:	Jumlah total movie unik. Menentukan ukuran embedding matrix V.
- embedding_dim=5:	Jumlah dimensi pada latent factor (embedding) untuk user dan movie. Semakin besar, semakin kaya representasinya, tapi berisiko overfitting jika datanya kecil.
- init_stddev=0.1: Standar deviasi untuk inisialisasi nilai awal embedding. Digunakan dalam tf.random.normal. Nilai ini menentukan seberapa "acak" nilai awal vektornya.
- reg_lambda=1e-4: Koefisien regularisasi untuk mencegah overfitting. Digunakan untuk menghukum bobot yang terlalu besar pada U, V, b_u, b_i.
- lr=0.005: Learning rate (kecepatan belajar) untuk optimizer Adam. Ini mengontrol seberapa besar langkah update parameter pada setiap iterasi.

Selain itu, saat proses train berlangsung, terdapat juga beberapa parameter utama seperti:
- A_train: Sparse Tensor yang berisi data rating untuk training. Bentuknya [userId, movieId] -> rating.
- A_test: Sparse Tensor yang berisi data rating untuk testing. Digunakan untuk mengukur generalisasi model.
- num_iters=2500: Jumlah total iterasi training. Tiap iterasi melakukan 1 update pada parameter model menggunakan gradient descent.
- eval_interval=15: Setiap 15 iterasi, model akan menghitung evaluasi pada data training dan testing. Nilai ini disimpan ke history dan dicetak ke layar.
- plot: True	Menentukan apakah akan ditampilkan plot evaluasi terhadap iterasi di akhir training. Berguna untuk melihat tren overfitting atau konvergensi.
  
## Evaluation
Untuk mengevaluasi kinerja kedua model rekomendasi, digunakan beberapa metrik berikut:
- Cosine Similarity (Mengukur kemiripan antara dua vektor dalam ruang berdimensi tinggi berdasarkan sudut di antara mereka.)
- Diversity (Keberagaman)-Novelty (Kebaruan) saat item rekomendasi muncul. Diversity akan mengukur seberapa beragam item yang direkomendasikan (tidak semuanya mirip satu sama lain) dan Novelty akan mengukur seberapa tidak populer / belum dikenal item yang direkomendasikan. Umumnya dihitung dari frekuensi item muncul di data.

Formula untuk Cosine Similarity yaitu:
![CS](https://raw.githubusercontent.com/hardiantots/recommendation-sys-ml10m/main/assets_rs/cosine_similarity.png)
\[
\text{sim}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \cdot \sqrt{\sum_{i=1}^{n} B_i^2}}
\]

- \( A, B \): dua vektor fitur (misalnya genre, embedding, dll)
- \( A \cdot B \): dot product dari kedua vektor
- \( \|A\| \): panjang (norma) dari vektor A

Nilai cosine similarity berkisar antara:

1 → sangat mirip (arah vektor sama)

0 → tidak mirip (vektor ortogonal)

-1 → sangat berlawanan arah (jarang muncul di sistem rekomendasi)

<br>

Formula untuk Diversity dan Novelty yaitu:
![Diver](https://raw.githubusercontent.com/hardiantots/recommendation-sys-ml10m/main/assets_rs/diversity.png)
\[
\text{Diversity}(S) = \frac{2}{|S|(|S|-1)} \sum_{i=1}^{|S|} \sum_{j=i+1}^{|S|} (1 - \text{sim}(i, j))
\]

- \( S \): himpunan item yang direkomendasikan
- \( \text{sim}(i, j) \): similarity antar item i dan j (biasanya cosine similarity)
- Semakin tinggi nilai Diversity, semakin beragam item rekomendasinya

![Novel](https://raw.githubusercontent.com/hardiantots/recommendation-sys-ml10m/main/assets_rs/novelty.png)
\[
\text{Novelty}(R) = \frac{1}{|R|} \sum_{i \in R} -\log_2(P(i))
\]

- \( R \): himpunan item yang direkomendasikan
- \( P(i) \): probabilitas/popularitas item i (misalnya: proporsi user yang menonton)
- Semakin tinggi nilai novelty, semakin unik/tidak umum item tersebut

<br>

Evaluasi untuk hasil rekomendasi dengan Content-Based Filtering yaitu:
![CS_Result](https://raw.githubusercontent.com/hardiantots/recommendation-sys-ml10m/main/assets_rs/cbased_result.png)
  - Untuk rekomendasi pada user yang menonton Toy Story, hasilnya menunjukkan bahwa rekomendasinya memiliki genre yang hampir identik "Adventure Animation Children Comedy Fantasy". Selain itu, judul-judul film berasal dari studio dan gaya yang serupa (misal: Pixar, DreamWorks, Studio Ghibli).
  - Jika melihat nilai Cosine Similarity, nilai kesamaannya sangat tinggi (mendekati 1), sehingga jika melihat juga ke diversitynya, rekomendasi ini kurang beragam namun cocok untuk bisa mempertahankan relevansi.
  - Jika melihat dari sisi Novelty, rekomendasi ini kurang memberikan kebaruan. Cenderung merekomendasikan film yang sudah sangat dikenal user.

Evaluasi untuk hasil rekomendasi dengan Collaborative Filtering yaitu:
![CB_Result1](https://raw.githubusercontent.com/hardiantots/recommendation-sys-ml10m/main/assets_rs/cb1.png)
  - Untuk rekomendasi berdasarkan preferensi user, diupayakan untuk menyesuaikan hasil rekomendasi dengan kecenderungan genre yang telah ditonton oleh user tertentu. Jika melihat nilai Cosine Similaritynya, hasilnya menunjukkan bahwa sebagian besar rekomendasinya masih sesuai dengan preferensi user,namun beberapa film horror/sci-fi masuk padahal itu bukan preferensi utama user.
  - Jika melihat nilai Novelty, terdapat judul-judul seperti Zombie Lake, Zombie Strippers!, dan The Onion Movie adalah film kurang terkenal di kalangan umum. Sehingga ini bisa untuk merekomendasikan film yang jarang muncul, memberi pengalaman baru.
  - Jika melihat dari sisi Diversity, beberapa film menyertakan kombinasi genre unik (e.g. Comedy Horror, Comedy Documentary Romance), yang menandakan bahwa nilai diversity cukup tinggi. Cocok untuk user yang suka eksplorasi.

![CB_Result2](https://raw.githubusercontent.com/hardiantots/recommendation-sys-ml10m/main/assets_rs/cb2.png)
  - Rekomendasi dilakukan berdasarkan preferensi film yang telah ditonton. Jika melihat nilai Cosine Similaritynya, hasilnya menunjukkan bahwa sebagian besar rekomendasinya banyak yang tidak sesuai dengan genre film Toy Story dan nilainya tergolong sangat rendah.
  - Jika melihat nilai Novelty dan Diversity, nampaknya sangat tinggi karena memberikan rekomendasi film yang kurang dikenal pengguna berdasarkan film yang telah ditonton.

Kesimpulan akhir:
- Dengan menggunakan Content-Based Filtering, hasilnya sangat genre-driven dan relevan dengan film asal (misalnya Toy Story → film anak-anak, animasi, dll) yang bisa digunakan untuk tetap menjaga relevasi konten yang ditonton pengguna, tetapi kurang bisa memberikan variasi rekomendasi (novelty dan diversitynya rendah) yang lebih terhadap user.
- Dengan menggunakan Collaborative Filtering, hasilnya dapat menemukan hubungan tersembunyi antar item dan antar user serta bisa merekomendasikan hal di luar genre yang kelihatan. Namun, untuk memberikan rekomendasi berdasarkan film yang telah ditonton, kurang memberikan hasil yang baik meskipun telah memberikan keragaman dalam rekomendasi filmnya
- Content-Based cocok untuk genre match, interpretasi eksplisit, dan ketika kita tahu apa yang mirip. Collaborative Filtering unggul dalam menemukan pola tersembunyi yang tidak bisa dilihat dari metadata saja, serta memberikan novelty dan diversity lebih tinggi.

## Daftar Pustaka
[[1]]Department of Computer Science, California State Polytechnic University, Pomona, CA, USA, Salloum, S., & Rajamanthri, D. (2021). Implementation and Evaluation of Movie Recommender Systems Using Collaborative Filtering. Journal of Advances in Information Technology, 12(3). https://doi.org/10.12720/jait.12.3.189-196

[[2]]Rakesh, S. (2023). Movie Recommendation System Using Content Based Filtering. Al-Bahir Journal for Engineering and Pure Sciences, 4(1). https://doi.org/10.55810/2313-0083.1043
