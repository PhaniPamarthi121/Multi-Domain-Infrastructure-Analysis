# Multi-Domain-Infrastructure-Analysis
Multi-domain analysis integrating pavement, weather, and text data using Spark, MLlib, and NLP to predict infrastructure performance and detect anomalies.

This project integrates pavement experiment logs, weather data, and burnout-related text to build a multi-domain analytics pipeline for infrastructure performance and human-centric insights.

## Project Overview

- Combined **pavement**, **weather**, and **burnout survey text** into a unified analysis workflow.
- Built a scalable analytics pipeline using **Python / PySpark / Spark**.
- Applied:
  - Regression and classification models for pavement performance prediction
  - Clustering and anomaly detection (e.g., Isolation Forest) for outlier detection
  - Basic NLP (TF-IDF, sentiment analysis) on burnout-related text
- Generated dashboards and visualizations to help infrastructure stakeholders make faster, data-driven decisions.

## Tech Stack

- **Languages:** Python, SQL (and/or PySpark SQL)
- **Big Data / ML:** Apache Spark, PySpark, Spark MLlib
- **NLP:** TF-IDF, sentiment analysis
- **Visualization / App:** Streamlit, Matplotlib / Seaborn
- **Data:** Pavement condition logs, weather records, and burnout survey text

## Repository Structure

```text
multi-domain-infrastructure-analysis/
│── README.md
│── notebooks/
│   └── multi_domain_analysis.ipynb.ipynb
│── reports/
│   ├── Multi_Domain_Project_Report.pdf
│   └── multi_domain_analysis.html
│── docs/
│   └── project_overview_team8.txt
│── src/
│   └── (optional scripts)
│── data/
│   └── (optional sample data)
│── requirements.txt
└── LICENSE
