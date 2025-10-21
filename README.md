# AI-Based-Lung-Disease-Detection-via-Sound
Here’s a polished **README.md** file for your **LSTM-based Lung Sound Classification Project** — formatted professionally and ready to include in your GitHub or project submission:

---


https://github.com/user-attachments/assets/a284ffb8-c21d-4c9d-8ce8-a1a655156fe0


# 🫁 Lung Sound Classification using LSTM

This project focuses on classifying lung sound recordings into various respiratory conditions using a **Long Short-Term Memory (LSTM)** deep learning model. The system is designed to assist in **early detection of respiratory diseases** by analyzing lung sound signals collected from digital stethoscopes or publicly available datasets.

---

## 📊 3.9 Dataset and Exploratory Data Analysis (EDA)

### 📁 Dataset Collection

* **Source:** [PhysioNet Respiratory Sound Database](https://physionet.org/) [15]
* **Total recordings:** 8,000
* **Categories:**

  * URTI (Upper Respiratory Tract Infection)
  * Healthy
  * Asthma
  * LRTI (Lower Respiratory Tract Infection)
  * Bronchiectasis
  * Pneumonia
  * Bronchiolitis
* **Average length per recording:** ~10 seconds
* **Epochs:** 125

### 📊 Exploratory Data Analysis (EDA)

* Visualized class distribution to detect imbalance
* Analyzed signal duration variability (0.2 s – 16.2 s)
* Inspected spectrograms and frequency components
* Verified data quality, removed noisy/outlier recordings

---

## 🧠 3.10 Model Selection

We selected a **Long Short-Term Memory (LSTM)** neural network because of its strength in modeling **sequential data** such as audio signals. LSTM solves the vanishing gradient problem common in traditional RNNs, making it ideal for **time-series data like lung and heart sounds**.

* Previous studies using hybrid **CNN–LSTM** models reported up to **99% accuracy** on datasets like ICBHI.
* Pure **LSTM** approaches achieved around **94% accuracy**, showing strong performance with lower complexity.
* LSTM thus offers an excellent balance between **accuracy, interpretability, and simplicity**.

---

## 🏋️‍♂️ 3.11 Model Training

The model was trained on a curated dataset of **5,000+ labeled recordings** representing various respiratory diseases such as Asthma, COPD, Pneumonia, and Bronchitis.

### 🔧 3.11.1 Preprocessing

* Noise filtering using **Butterworth filters** (cutoffs: 0–10 Hz and >4,000 Hz)
* Separation of **heart and lung signals**

### 🔪 3.11.2 Segmentation

* Audio split into **respiratory cycles** (~2.7 s average, range: 0.2–16.2 s)

### 🎯 3.11.3 Feature Extraction

Extracted features to capture both time and frequency domain characteristics:

* **MFCCs** (13–30 coefficients)
* **Chroma features**
* **Spectral centroid & bandwidth**
* **Short-Time Fourier Transform (STFT)**
* **Wavelet scalograms** (for additional experiments)

### 📈 3.11.4 Data Augmentation

* Increased samples in minority classes: URTI, Healthy, Asthma, LRTI, Bronchiectasis
* Applied pitch shifting, time stretching, and background noise injection

### ⚙️ 3.11.5 Training Setup

| Parameter        | Value                        |
| ---------------- | ---------------------------- |
| Optimizer        | Adam                         |
| Loss Function    | Categorical Cross-Entropy    |
| Epochs           | 50–100 (with Early Stopping) |
| Batch Size       | 32                           |
| Train/Test Split | 80/20                        |

The training process allowed the LSTM model to learn distinct respiratory patterns and classify anomalies effectively.

---

## 📊 3.12 Model Evaluation

The model’s performance was thoroughly evaluated to determine its accuracy and reliability in classifying respiratory conditions such as URTI, Asthma, LRTI, Bronchiectasis, and Healthy cases.

---

## 📏 3.13 Evaluation Metrics

| Metric                   | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| **Accuracy**             | Measures overall correct predictions                     |
| **Precision**            | Fraction of relevant instances among retrieved instances |
| **Recall (Sensitivity)** | Ability to correctly identify positive cases             |
| **F1-Score**             | Harmonic mean of precision and recall                    |
| **Confusion Matrix**     | Visualizes actual vs. predicted class distribution       |

---

## 🔁 3.14 Cross-Validation

To ensure **robustness and generalization**, we applied **k-fold cross-validation** (typically k = 5 or 10). This reduced overfitting risk and ensured consistent performance across different splits of the dataset.

---

## 🧪 3.15 Real-Time Testing

In addition to offline dataset evaluation, the model was tested with **real-time lung sound recordings** captured using a **digital stethoscope**.

* The LSTM model provided **instant classification results**.
* Predictions were verified against labeled ground-truth data.
* Demonstrated reliable real-time performance suitable for **clinical and mobile healthcare applications**.

---

## 📁 Project Structure

```
├── data/
│   ├── recordings/          # Raw lung sound audio files
│   ├── spectrograms/        # Spectrogram images (optional)
│   └── labels.csv           # Class labels
├── src/
│   ├── preprocess.py        # Preprocessing and filtering
│   ├── feature_extraction.py# MFCC/STFT feature extraction
│   ├── model.py             # LSTM model definition
│   ├── train.py             # Training script
│   └── evaluate.py          # Metrics and evaluation
├── notebooks/
│   └── EDA.ipynb            # Exploratory Data Analysis
├── requirements.txt
├── README.md
└── main.py                  # Entry point for real-time testing
```

---

## 🧰 Requirements

Install required dependencies:

```bash
pip install -r requirements.txt
```

Key libraries:

 `numpy`*
 `pandas`
 `librosa`
 `tensorflow` / `keras`
 `matplotlib`
 `scikit-learn`

---

## 🚀 How to Run

### 1. Preprocess and extract features:

```bash
python src/preprocess.py
python src/feature_extraction.py
```

### 2. Train the LSTM model:

```bash
python src/train.py
```

### 3. Evaluate the model:

```bash
python src/evaluate.py
```

### 4. Real-time testing (with Bluetooth stethoscope):

```bash
python main.py
```

---

## 📊 Results
Accuracy:** ~94% – 97% (dataset-based)
F1-Score:** ~0.94 (balanced across classes)
Real-time Accuracy:** ~92% (verified with digital stethoscope data)

---

## 📚 References

[15] PhysioNet Respiratory Sound Database. Available at: [https://physionet.org](https://physionet.org)

---

## 🩺 Future Work

* Integrating a **hybrid CNN–LSTM** model to capture spatial + temporal features
* Expanding dataset with more disease categories
* Deploying as a **mobile health app** for real-time clinical use

