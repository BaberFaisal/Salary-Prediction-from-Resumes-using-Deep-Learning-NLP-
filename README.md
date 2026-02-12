# Salary Prediction from Resumes using Deep Learning (NLP)

This repository contains an implementation of a **Natural Language Processing (NLP)** and **Deep Learning** pipeline for predicting job salaries based on resumes and job descriptions.  


---

## Project Overview

The goal of this project is to predict the **log-transformed normalized salary (`Log1pSalary`)** of job postings using both **textual** and **categorical** data features.  
Key text features include the **job title** and **full job description**, while categorical features include the company name, contract type, and location.

The project demonstrates:
- NLP preprocessing (tokenization, vocabulary creation)
- Feature engineering for text and categorical data
- Neural network design and training using **PyTorch**
- Model evaluation and visualization
- Model interpretability via token-level contribution analysis

---

## Dataset

The dataset used is from the **Kaggle Job Salary Prediction** competition:
🔗 [https://www.kaggle.com/c/job-salary-prediction/data](https://www.kaggle.com/c/job-salary-prediction/data)

**Main file:** `Train_rev1.zip`

Each record contains:
- `Title`: Job title  
- `FullDescription`: Complete job posting text  
- `Category`: Job category  
- `Company`: Company name  
- `LocationNormalized`: Job location  
- `ContractType` and `ContractTime`: Type and duration  
- `SalaryNormalized`: Target variable (normalized salary)

The log transformation `Log1pSalary = log(1 + SalaryNormalized)` is used to stabilize variance and handle skewness.

---


