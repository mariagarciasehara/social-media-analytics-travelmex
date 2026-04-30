# instagram-analytics-travelmex
Data Science project — Instagram performance analytics for a travel agency (EDA, Hypothesis Testing, ML)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Lab-orange?logo=jupyter)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-red?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Platforms](https://img.shields.io/badge/Platforms-Instagram%20%7C%20Facebook%20%7C%20TikTok%20%7C%20YouTube-purple)

> **End-to-end Data Science project** analyzing Instagram performance for a Miami-based travel agency.  
> From raw CSV exports to machine learning predictions — actionable insights that drive real business decisions.

---

## 🎯 Business Problem

Travel Mex Tours is a small travel agency in Miami with **2,608 Instagram followers** looking to grow their social media presence.  
The agency needed answers to three key questions:

1. **When** should they post to maximize engagement?
2. **What type of content** performs best?
3. Can we **predict engagement before publishing** to optimize content strategy?

---

## 📈 Key Results

| Metric | Travel Mex | Industry Benchmark (1k–10k accounts) |
|--------|-----------|--------------------------------------|
| Avg Engagement Rate | **15.4%** | 3–6% |
| Performance vs Benchmark | **3× above average** ✅ | — |
| Best Content Type | Reels | — |
| Best Day to Post | Tuesday | — |
| ML Model Accuracy | **R² = 0.88** | — |
| Avg Prediction Error | **±2.17%** | — |

---

## 🤖 Machine Learning Results

Four models were trained and compared using 5-fold cross-validation:

| Model | R² Score | MAE | CV R² |
|-------|----------|-----|-------|
| **Gradient Boosting** ⭐ | **0.8834** | **±2.17%** | **0.8692** |
| Random Forest | 0.8701 | ±2.31% | 0.8541 |
| Ridge Regression | 0.7203 | ±3.14% | 0.7089 |
| Linear Regression | 0.6987 | ±3.28% | 0.6812 |

**Top 3 drivers of engagement:**
1. 🥇 **Reach** (48.5% importance) — wider reach = higher engagement
2. 🥈 **Likes** (28.3% importance) — early likes trigger the algorithm
3. 🥉 **Views** (5.3% importance) — video consumption drives interaction

---

## 🧪 Hypothesis Testing Results

Three statistical tests were run at α = 0.05:

| Test | Question | Result |
|------|----------|--------|
| One-Way ANOVA | Does content type affect ER? | Significant ✅ |
| One-Way ANOVA | Does Reel duration affect ER? | Not significant ❌ |
| Welch t-test | Do weekend posts outperform weekdays? | Not significant ❌ |

> **Key insight:** Content type matters statistically — but duration and posting day don't.  
> Focus on **content quality and format** over timing optimization.

---

## 📁 Project Structure

```
social-media-analytics-travelmex/
│
├── notebooks/
│   └── instagram/
│       ├── 01_EDA.ipynb                  ← Exploratory Data Analysis
│       ├── 02_Hypothesis_Testing.ipynb   ← Statistical Tests
│       └── 03_Machine_Learning.ipynb     ← ML Prediction Models
│
├── src/
│   ├── load_data.py          ← Robust CSV loader (any Instagram export)
│   ├── preprocessing.py      ← Feature engineering + metadata
│   ├── eda.py                ← EDA functions + chart generation
│   ├── hypothesis_testing.py ← ANOVA + t-test pipeline
│   ├── machine_learning.py   ← ML training + evaluation
│   └── app.py                ← Streamlit dashboard
│
├── data/
│   ├── instagram/            ← Instagram CSV exports (gitignored)
│   ├── facebook/             ← Facebook CSV exports (gitignored)
│   ├── tiktok/               ← TikTok CSV exports (gitignored)
│   └── youtube/              ← YouTube CSV exports (gitignored)
│
├── outputs/
│   └── figures/              ← Generated charts (gitignored)
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Statistics | SciPy |
| Machine Learning | Scikit-learn (RF, GBM, Ridge, Linear) |
| Dashboard | Streamlit |
| Environment | Jupyter Lab |
| Version Control | Git + GitHub |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/mariagarciasehara/social-media-analytics-travelmex.git
cd social-media-analytics-travelmex
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Instagram CSV
Export your data from Instagram Insights and place the CSV in:
```
data/instagram/your_export.csv
```
The notebooks automatically detect the most recent CSV — no code changes needed.

### 4. Run the notebooks in order
```bash
jupyter lab
```
Open `notebooks/instagram/` and run:
1. `01_EDA.ipynb`
2. `02_Hypothesis_Testing.ipynb`
3. `03_Machine_Learning.ipynb`

---

## 📊 Notebooks Overview

### 01 — Exploratory Data Analysis
- Engagement rate distribution vs industry benchmark
- Best day and hour to post (with statistical backing)
- Content type performance comparison
- Reach vs engagement correlation
- Monthly trend analysis
- Top 10 best performing posts

### 02 — Hypothesis Testing
- **Test 1:** One-Way ANOVA — Does content type affect engagement?
- **Test 2:** ANOVA — Does Reel duration affect engagement?
- **Test 3:** Welch t-test — Do weekend posts outperform weekdays?
- Assumption checks: Shapiro-Wilk (normality) + Levene (equal variance)
- Effect size (η²) for all tests

### 03 — Machine Learning
- Feature selection and engineering
- 4 models trained + compared
- 5-fold cross-validation
- Feature importance analysis
- Actual vs predicted visualization
- **🚀 Live engagement predictor** — input a post's details and predict ER before publishing

---

## 🗺️ Roadmap

- [x] Instagram EDA
- [x] Hypothesis Testing
- [x] Machine Learning Prediction
- [ ] Streamlit Dashboard (in progress)
- [ ] Content Strategy Notebook (clustering + hashtag analysis)
- [ ] Time Series Forecasting (Prophet)
- [ ] Automated PDF Report
- [ ] Facebook Analysis
- [ ] TikTok Analysis
- [ ] YouTube Analysis
- [ ] Multi-platform Comparison Dashboard

---

## 👩‍💻 Author

**Maria Garcia Sehara**  
Data Analyst | Miami, FL  
[GitHub](https://github.com/mariagarciasehara)

---

## 📄 License

This project is for educational and portfolio purposes.  
Data belongs to Travel Mex Tours and is not included in this repository.

