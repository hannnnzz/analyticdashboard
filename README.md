# Yelp Analytics Dashboard

> An end-to-end analytics platform built on the Yelp Open Dataset, combining Business Intelligence, Natural Language Processing, and Machine Learning in a single interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=flat-square&logo=plotly)

---

## Modules

| Module | Description |
|--------|-------------|
| **Business Intelligence** | Market overview, reputation & popularity analysis, differentiation & strategy |
| **NLP — Sentiment & Emotion** | Review-level sentiment scoring and emotion classification |
| **NLP — Business Summary** | Automated business summaries generated from review data |
| **User Segmentation** | KMeans & HDBSCAN clustering, social graph visualization, user prediction |
| **Churn Analysis** | User churn prediction and check-in behavior analysis |
| **Recommendation System** | Personalized business recommendations using Hybrid Neural Collaborative Filtering (GMF + MLP + Content) |

---

## Dataset

This project uses the **[Yelp Open Dataset](https://www.yelp.com/dataset)** and is included in the repository.

| File | Used In |
|------|---------|
| `BusinessData.xlsx` | BI, RecSys |
| `ReviewDataNew.xlsx` | NLP Sentiment, Emotion |
| `UserUniqueReview.xlsx` | Churn, Clustering |
| `CheckinData.xlsx` | Churn Checkin |
| `sampled-userdata.json` | User loader |
| `sampled_tipdata.json` | Tip loader |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/yelp-analytics-dashboard.git
cd yelp-analytics-dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py --server.fileWatcherType none
```

> `--server.fileWatcherType none` is required to avoid a known conflict between Streamlit's file watcher and PyTorch.

---

## Models

All pre-trained models are stored in `models/` and loaded at startup.

| Model | Type | Location |
|-------|------|----------|
| Sentiment & Emotion | Pre-computed DataFrame (`.pkl`) | `models/sentimo/` |
| Business Summary | Pre-computed summaries (`.pkl`) | `models/summary/` |
| User Clustering | KMeans + HDBSCAN + Scaler (`.pkl`) | `models/clustering/` |
| Churn Prediction | Scikit-learn classifier (`.pkl`) | `models/churn/` |
| Recommendation | Hybrid NCF — PyTorch (`.pt`) + encoders | `models/recsys/` |

> **Note:** The Recommendation System model is currently in a demonstrative state and has not been fully optimized. Results shown are for exploration purposes and may not reflect final performance.

---

## Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Visualization:** [Plotly](https://plotly.com/python/)
- **Deep Learning:** [PyTorch](https://pytorch.org/)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/)
- **Data Processing:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)

---

## License

This project is for academic and educational purposes.  
Copyright (c) 2025 Adhystira Raihannoeza.  
Yelp dataset usage is subject to [Yelp's Dataset Terms of Use](https://www.yelp.com/dataset/documentation/main).