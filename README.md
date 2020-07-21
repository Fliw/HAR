 # Rubix ML - Pendeteksi Pergerakan Manusia
 ## Artificial Intelligence based on Softmax Classifier
Contoh proyek yang menunjukkan masalah pengenalan aktivitas manusia (HAR) menggunakan data sensor ponsel yang direkam dari unit pengukuran inersia internal (PII). Data pelatihan adalah pembacaan sensor beranotasi manusia dari 30 relawan sambil melakukan berbagai tugas seperti duduk, berdiri, berjalan, dan berbaring. Setiap sampel berisi Foto 561 Keadaan, namun, kami menunjukkan bahwa dengan teknik yang disebut *proyeksi acak* kami dapat mengurangi dimensi tanpa kehilangan keakuratan. Trainer yang akan kami latih untuk menyelesaikan tugas ini adalah Algoritma [Softmax Classifier](https://docs.rubixml.com/en/latest/classifiers/softmax-classifier.html) yang merupakan generalisasi multiclass dari classifier biner [Logistic Regression](https://docs.rubixml.com/en/latest/classifiers/logistic-regression.html).

- **Kesulitan**: Medium
- **Waktu Untuk Belajar**: Menit

## Instalasi
Clone Project ini menggunakan [Composer](https://getcomposer.org/):
```sh
$ composer create-project rubix/har
```

## Kebutuhan
- [PHP](https://php.net) 7.2 Atau Diatasnya

#### Rekomendasi
- [Ekstensi Tensor](https://github.com/RubixML/Tensor) Buat Latih dan ambil kesimpulan lebih cepat.
- Minimum 1GB RAM Digunakan untuk training.

## Tutorial

### Pengenalan
Percobaan telah dilakukan dengan sekelompok 30 sukarelawan dalam kurun usia 19-48 tahun. Setiap orang melakukan enam aktivitas (berjalan, berjalan menaiki tangga, berjalan menuruni tangga, duduk, berdiri, dan berbaring) mengenakan smartphone di pinggang mereka. Dengan menggunakan accelerometer dan gyroscope yang tertanam, akselerasi linear 3-aksial dan kecepatan sudut 3-aksial dicatat pada kecepatan konstan 50Hz. Sinyal-sinyal sensor pra-diproses dengan menerapkan filter noise dan kemudian sampel dalam jendela geser lebar tetap 2,56 detik. Tujuan kami adalah untuk membangun classifier untuk mengenali aktivitas mana yang dilakukan pengguna dengan memberikan beberapa data yang tidak terlihat.

> **Catatan:** Kode sumber untuk contoh ini dapat ditemukan di [train.php](https://github.com/fliw/HAR/blob/master/train.php) Di Folder Root.

### Ekstraksi Data
Data diberikan kepada kami dalam dua file NDJSON (newline delimited JSON) di dalam Folder root. Satu file berisi sampel pelatihan dan yang lainnya untuk pengujian. Kami akan menggunakan [NDJSON](https://docs.rubixml.com/en/latest/extractors/ndjson.html) extractor disediakan dalam Rubix ML untuk mengimpor data pelatihan ke objek datasets yang baru [Labeled](https://docs.rubixml.com/en/latest/datasets/labeled.html). Karena extractor adalah iterator, kita dapat meneruskan extractor langsung ke Factory Method `fromIterator ()`. 

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;

$dataset = Labeled::fromIterator(new NDJSON('train.ndjson'));
```

### Persiapan Datasets
Dalam Machine Learning, pengurangan dimensi sering digunakan untuk mengompres sampel input sehingga sebagian besar atau semua informasi dipertahankan. Dengan mengurangi jumlah fitur input, kami dapat mempercepat proses pelatihan. (Semakin Banyak Semakin Pintar dan Akurat) [Random Projection](https://en.wikipedia.org/wiki/Random_projection) adalah teknik reduksi dimensionalitas tanpa pengawasan yang efisien secara komputasional berdasarkan pada [Johnson-Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma) yang menyatakan bahwa satu set titik dalam ruang berdimensi tinggi dapat disematkan ke dalam ruang berdimensi lebih rendah sedemikian rupa sehingga jarak antara titik hampir terpelihara. Untuk menerapkan pengurangan dimensi pada dataset HAR, kami akan menggunakan Proyektor [Gaussian Random Projector](https://docs.rubixml.com/en/latest/transformers/gaussian-random-projector.html) sebagai bagian dari pipeline kami. Gaussian Random Projector menerapkan transformasi linear acak yang diambil dari distribusi Gaussian ke matriks sampel. Kami akan menetapkan jumlah target dimensi ke 110 yang kurang dari 20% dari dimensi input asli.

Terakhir, kami akan memusatkan dan mengukur dataset menggunakan [Z Scale Standardizer](https://docs.rubixml.com/en/latest/transformers/z-scale-standardizer.html) bahwa nilai-nilai fitur memiliki 0 mean dan varians unit. Langkah terakhir ini akan membantu trainer selesai dengan kesimpulan lebih cepat selama training.

Kami akan membungkus transformasi ini dalam sebuah [Pipeline](https://docs.rubixml.com/en/latest/pipeline.html) sehingga alat kelengkapan mereka dapat bertahan bersama dengan model.

### Memulai The Learner
Sekarang, kita akan mengalihkan perhatian kita ke pengaturan parameter trainer. [Softmax Classifier](https://docs.rubixml.com/en/latest/classifiers/softmax-classifier.html) adalah jenis jaringan saraf single layer dengan [Softmax](https://docs.rubixml.com/en/latest/neural-network/activation-functions/softmax.html) output layer. Pelatihan dilakukan secara iteratif menggunakan Mini Batch Gradient Descent di mana pada setiap parameter model mengambil langkah ke arah minimum dari kesalahan gradien yang dihasilkan oleh fungsi biaya yang ditentukan pengguna seperti [Cross Entropy](https://docs.rubixml.com/en/latest/neural-network/cost-functions/cross-entropy.html).

Parameter hyper-pertama Softmax Classifier adalah `ukuran batch` yang mengontrol jumlah sampel yang dimasukkan ke dalam jaringan pada suatu waktu. Ukuran bets diperdagangkan dari kecepatan pelatihan untuk kelancaran estimasi gradien. Ukuran batch 256 berfungsi cukup baik untuk contoh ini sehingga kami akan memilih nilai itu tetapi jangan ragu untuk bereksperimen dengan pengaturan lain dari ukuran batch Anda sendiri.

Parameter-hiper berikutnya adalah Gradient Descent `optimizer` dan` learning rate` terkait. Itu [Momentum](https://docs.rubixml.com/en/latest/neural-network/optimizers/momentum.html) optimizer adalah optimizer adaptif yang menambah kekuatan momentum untuk setiap pembaruan parameter. Momentum membantu mempercepat pelatihan dengan melintasi gradien lebih cepat. Ini menggunakan tingkat pembelajaran global yang dapat diatur oleh pengguna dan biasanya berkisar antara 0,1 hingga 0,0001. Pengaturan default 0,001 berfungsi dengan baik untuk contoh ini sehingga kami akan membiarkannya.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\GaussianRandomProjector;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel(
    new Pipeline([
        new GaussianRandomProjector(110),
        new ZScaleStandardizer(),
    ], new SoftmaxClassifier(256, new Momentum(0.001))),
    new Filesystem('har.model')
);
```

Kami akan membungkus seluruh pipeline dalam sebuah [Persistent Model](https://docs.rubixml.com/en/latest/persistent-model.html) meta-estimator yang menambahkan metode `save ()` dan `load ()` ke estimator basis. Model Persistent membutuhkan sebuah [Persister](https://docs.rubixml.com/en/latest/persisters/api.html) objek untuk mengatakan di mana menyimpan data model serial. [Filesystem](https://docs.rubixml.com/en/latest/persisters/filesystem.html) persister menyimpan dan memuat data model ke file yang terletak di jalur yang ditentukan pengguna dalam penyimpanan. 

### Mengatur Logger
Karena Softmax Classifier mengimplementasikan [Verbose](https://docs.rubixml.com/en/latest/verbose.html) antarmuka, kita bisa mencatat kemajuan pelatihan secara real-time. Untuk mengatur logger, masukkan sebuah [PSR-3](https://www.php-fig.org/psr/psr-3/) instance logger yang kompatibel dengan metode `setLogger ()` pada instance pelajar. [Screen](https://docs.rubixml.com/en/latest/other/loggers/screen.html) logger yang disertakan dengan Rubix ML adalah pilihan default yang bagus jika Anda hanya perlu sesuatu yang sederhana untuk di-output ke konsol.

```php
use Rubix\ML\Other\Loggers\Screen;

$estimator->setLogger(new Screen('HAR'));
```

### Training
mulailah melatih pelajar, panggil metode `train ()` pada instance dengan dataset pelatihan sebagai argumen.

```php
$estimator->train($dataset);
```

### Training Loss
Selama pelatihan, pelajar akan mencatat kehilangan pelatihan di setiap waktu yang dapat kita plot untuk memvisualisasikan kemajuan pelatihan. Kehilangan pelatihan adalah nilai dari fungsi biaya di setiap zaman dan dapat diinterpretasikan sebagai jumlah kesalahan yang tersisa dalam model setelah langkah pembaruan. Untuk mengembalikan array dengan nilai fungsi biaya pada setiap zaman, panggil metode `steps ()` pada trainer.

```php
$losses = $estimator->steps();
```

Ini adalah contoh plot garis fungsi biaya Entropi Silang dari sesi pelatihan. Seperti yang Anda lihat, model belajar dengan cepat selama zaman awal dengan pelatihan yang lebih lambat mendekati tahap akhir saat pelajar memperbaiki parameter model.

![Cross Entropy Loss](https://raw.githubusercontent.com/RubixML/HAR/master/docs/images/training-loss.svg?sanitize=true)

### Saving
Karena kita membungkus estimator dalam pembungkus Model Persisten, kita dapat menyimpan model dengan memanggil metode `save ()` pada instance estimator.

```php
$estimator->save();
```

Untuk menjalankan skrip pelatihan, panggil dari baris perintah seperti ini.
```sh
$ php train.php
```

### Cross Validation
Penulis dataset memberikan 2.947 sampel pengujian berlabel tambahan yang akan kami gunakan untuk menguji model. Kami telah mengadakan sampel ini sampai sekarang karena kami ingin dapat menguji model pada sampel yang belum pernah dilihat sebelumnya. Mulailah dengan mengekstraksi sampel pengujian dan label kebenaran dari file `test.ndjson`.

> **Note:** Kode sumber untuk contoh ini dapat ditemukan di [validate.php](https://github.com/RubixML/HAR/blob/master/validate.php) file folder root.

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;

$dataset = Labeled::fromIterator(new NDJSON('test.ndjson'));
```

### Load Model from Storage
Untuk memuat pipa estimator / transformator yang kami instantiated sebelumnya, panggil metode static `load ()` pada [Persistent Model](https://docs.rubixml.com/en/latest/persistent-model.html) kelas dengan instance Persister yang menunjuk ke model dalam penyimpanan.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('har.model'));
```

### Membuat Prediksi
Untuk mendapatkan prediksi dari model, berikan set pengujian ke metode `predict ()` pada instance estimator.

```php
$predictions = $estimator->predict($dataset);
```

### Generate report
Laporan validasi silang memberikan statistik terperinci tentang kinerja model yang diberi label ground-truth. [Multiclass Breakdown](https://docs.rubixml.com/en/latest/cross-validation/reports/multiclass-breakdown.html) laporan memecah kinerja model di tingkat kelas dan metrik output seperti akurasi, presisi, penarikan, dan banyak lagi. [Confusion Matrix](https://docs.rubixml.com/en/latest/cross-validation/reports/confusion-matrix.html) adalah tabel yang membandingkan label yang diprediksi dengan label yang sebenarnya untuk ditampilkan jika model mengalami kesulitan memprediksi kelas tertentu. Kami akan membungkus kedua laporan dalam [Aggregate Report](https://docs.rubixml.com/en/latest/cross-validation/reports/aggregate-report.html) sehingga kami dapat menghasilkan kedua laporan sekaligus.

```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);
```

Now, buat laporan menggunakan prediksi dan label dari set pengujian.

```php
$results = $report->generate($predictions, $dataset->labels());
```

Untuk menjalankan skrip validasi, masukkan perintah berikut di prompt perintah.
```php
$ php validate.php
```

Output dari laporan harus terlihat seperti output di bawah ini. Kerja bagus! Seperti yang Anda lihat, penaksir kami adalah sekitar 97% akurat dan memiliki spesifisitas yang sangat baik dan nilai prediksi negatif.

```json
[
    {
        "overall": {
            "accuracy": 0.9674308943546821,
            "precision": 0.9063809316861989,
            "recall": 0.9048187793615003,
            "specificity": 0.9802554195397294,
            "negative_predictive_value": 0.9803712249716344,
            "false_discovery_rate": 0.09361906831380108,
            "miss_rate": 0.09518122063849947,
            "fall_out": 0.019744580460270538,
            "false_omission_rate": 0.01962877502836563,
            "f1_score": 0.905257137386163,
            "mcc": 0.8858111380161123,
            "informedness": 0.8850741989012301,
            "markedness": 0.8867521566578332,
            "true_positives": 2675,
            "true_negatives": 13375,
            "false_positives": 272,
            "false_negatives": 272,
            "cardinality": 2947
        },
    }

]
```

### Next Steps
Sekarang setelah Anda menyelesaikan tutorial tentang mengklasifikasikan aktivitas manusia menggunakan Softmax Classifier, lihat apakah Anda dapat mencapai hasil yang lebih baik dengan menyempurnakan beberapa parameter hiper. Lihat seberapa besar reduksi dimensi memengaruhi akurasi akhir estimator dengan mengeluarkan Gaussian Random Projector dari pipa. Apakah ada teknik pengurangan dimensi lain yang bekerja lebih baik?

## Original Dataset
Contact: Jorge L. Reyes-Ortiz (1,2), Davide Anguita (1), Alessandro Ghio (1), Luca Oneto (1) and Xavier Parra (2) Institutions: 1 - Smartlab - Non-Linear Complex Systems Laboratory Laboratory DITEN - University of Genoa, Genoa (I-16145), Italy. 2 - CETpD - Technical Research Center for Dependency Care and Autonomous Living Polytechnic University of Catalonia (BarcelonaTech). Vilanova i la Geltrú (08800), Spain activityrecognition '@' smartlab.ws

## References:
>- Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra dan Jorge L. Reyes-Ortiz. Kumpulan Data Domain Publik untuk Pengakuan Aktivitas Manusia Menggunakan Ponsel Cerdas. Simposium Eropa ke-21 tentang Jaringan Syaraf Tiruan, Inteligensi Komputasi dan Pembelajaran Mesin, ESANN 2013. Bruges, Belgia 24-26 April 2013.

## License
Kode ini dilisensikan [MIT](LICENSE.md) dan tutorialnya berlisensi [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).