# 📉 Customer Churn Prediction using ANN

![Model Architecture](images/Churn.PNG)

This project uses an **Artificial Neural Network (ANN)** to predict customer churn for a bank. The model determines whether a customer is likely to leave the bank (churn) based on personal and account-level features such as age, tenure, balance, etc.

---

## 🚀 Project Overview

Customer churn prediction is a critical task in the banking and telecom industries, enabling businesses to proactively retain valuable customers. In this project, we:

- Preprocessed a customer dataset (encoding, scaling, etc.)
- Built and trained an **ANN** using TensorFlow/Keras
- Evaluated the model's performance
- Created a **Streamlit app** to make real-time churn predictions

---

## 🛠️ Tools & Technologies

| Tool           | Logo |
|----------------|------|
| Python         | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| Pandas         | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) |
| NumPy          | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) |
| Scikit-learn   | ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) |
| TensorFlow     | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) |
| Streamlit      | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) |
| Jupyter Notebook | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white) |
| Matplotlib     | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white) |

---

## 🧠 Model Architecture

- **Input Layer**: Encoded and scaled customer data
- **Hidden Layers**: Two Dense layers with ReLU activation
- **Output Layer**: Single neuron with Sigmoid activation (binary classification)

---

## 📊 Dataset

- **Source**: Bank Customer Churn Dataset (CSV format)
- **Features**: Credit Score, Geography, Gender, Age, Tenure, Balance, Num of Products, Has Credit Card, Is Active Member, Estimated Salary
- **Target**: `Exited` (1 = churned, 0 = stayed)

---

## 📈 Model Performance

- **Training Accuracy**: ~86%
- **Validation Accuracy**: ~84%
- **Loss**: ~0.35

The ANN model demonstrates reliable performance for identifying churned customers, providing banks with insight to target retention efforts.

---

## 🌐 Streamlit App

![Churn Streamlit](images/streamlit-app.png)

The app allows users to input customer details and receive real-time churn predictions.

---

## 🖥️ Installation & Usage

### 🔹 Clone the repository:

```bash
git clone https://github.com/achraf-bogryn/ANN-Classification-Probelm.git
cd ANN-Classification-Probelm.
```

### 🔹 Install the required packages:
```bash
pip install -r requirements.txt
```

### 🔹 Run the Streamlit App:
```bash
cd apps
streamlit run app.py
```

### 📦 Project Structure
```bash 
sentiment-analysis-imdb/
├── Ann                                        # Notebooks containing Training and preprocessing data 
├── apps/app.py                                # Streamlit interface
├── models/                                    # model and all transformation 
├── images/                                    # Visual assets for README
├── dataset/                                   # dataset
├── logs/                                      # monitoring train and test process 
├── requirements.txt                           # Dependencies
└── README.md
```

