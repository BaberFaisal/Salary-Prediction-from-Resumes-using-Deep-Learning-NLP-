## Deep Learning for Large-Scale Salary Prediction

This repository contains a deep learning project aimed at predicting job salaries based on resumes and job descriptions using a multi-modal neural network architecture.



##  Results Summary

The model was trained to minimize Mean Squared Error (MSE) on the log-transformed salary. Below are the performance metrics recorded during the training of the final architecture:

| Epoch | Validation MSE | Validation MAE |
|-------|----------------|----------------|
| 0     | 0.14216        | 0.28792        |
| 1     | 0.12233        | 0.26559        |
| 2     | 0.10758        | 0.24844        |
| 3     | 0.10229        | 0.24108        |
| 4     | 0.09672        | 0.23437        |

##  Key Methodology

### 1. Multi-Modal Architecture
The model processes three distinct types of input features:
- **Job Title:** Processed via an embedding layer and global average pooling.
- **Job Description:** Processed via a separate embedding branch.
- **Categorical Data:** (Category, Company, Location, etc.) One-hot encoded and projected through a linear layer.

The features from these branches are concatenated and passed through a Multi-Layer Perceptron (MLP) with **Dropout (0.2)** to prevent overfitting.

### 2. Handling Target Distribution
Salary data is famously "fat-tailed." To ensure effective optimization using MSE, we predicted the **Log1pSalary** instead of the raw dollar amount.

### 3. Model Explainability
We implemented a perturbation-based explanation method. By systematically dropping individual tokens from job titles and descriptions, we measured the "importance" of each word to the final salary prediction.

## Installation & Usage

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/salary-prediction.git
   ```
2. **Install requirements:**
   ```bash
   pip install torch nltk pandas numpy scikit-learn
   ```
3. **Data:** Download the Adzuna dataset (Train_rev1) and ensure it is in the project directory.

##  Credits
This project is based on a seminar from the Yandex Data School NLP course, originally developed by [Oleg Vasilev](https://github.com/Omrigan/).
