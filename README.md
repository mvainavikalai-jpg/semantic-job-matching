# Semantic Job Matching Using Transformer Models

## 📌 Overview
A comparative study of TF-IDF, BERT, and DistilBERT for classifying and matching job postings. This project demonstrates that while Transformer models offer better semantic understanding, DistilBERT provides the best balance between accuracy and computational cost.

## 🛠️ Tech Stack
- **Models:** TF-IDF (Baseline), BERT (109.5M params), DistilBERT (67M params)
- **Libraries:** HuggingFace Transformers, Scikit-learn, Pandas, PyTorch
- **Hardware:** CUDA GPU (Google Colab T4)

## 📊 Dataset
- **Source:** MotherDuck Public Parquet Files (Jan 2025 - Aug 2026)
- **Size:** 156,227 clean job postings (from 199,999 raw)
- **Classes:** 4 Job Categories (Analytics dominates at 62%, BI is the hardest at 5.8%)
- **Split:** 80% Train (124,981) / 20% Test (31,246)

## 🚀 Methodology & Results
- **TF-IDF:** Achieved 91.5% Accuracy, but showed high recall (88%) and low precision (61%) on the hardest class (BI), indicating it is "too generous".
- **BERT:** Highest accuracy at **92.7%**, with strong precision (89-93% range). However, it had the highest computational cost.
- **DistilBERT:** Achieved **92.5% accuracy** (retaining 99.8% of BERT's performance) while reducing training time by **48%** and inference time by **44%**.

## 💡 Key Takeaways
- **Speed vs. Accuracy:** DistilBERT provides the best trade-off, making it the most suitable choice for production environments where compute cost is a factor.
- **Semantic Matching:** Transformers successfully capture semantic meaning, improving the F1-score on the hardest class from 0.72 (TF-IDF) to 0.81 (Transformers).
