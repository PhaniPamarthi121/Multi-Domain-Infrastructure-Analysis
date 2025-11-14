
# README

## Project Title: Predictive Road and Pavement Analytics (AIT 614 Final Project)

## Team Members
- Dhanushi Panga
- Phani Satya Sai Pamarthi
- Raja Ruthvik Shetty
- Suraj Bharadwaj
- Mandava Venkata Yatish
- Chandra Nalla

## Objective
This project focuses on predictive modeling using multi-domain datasets comprising pavement condition, weather records, traffic trends, and textual reviews. The goal is to integrate structured and unstructured data to build regression and classification models, perform anomaly detection, and derive insights for civil infrastructure analytics.

## System Requirements
- Jupyter Notebook
- Python 3.8+
- Required Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost
  - nltk
  - sklearn
  - folium (for geo-based clustering visualization)

## Dataset Information
- Multiple CSV files including pavement metadata, weather logs, traffic logs, and review datasets.
- Ensure all CSV files are uploaded and available in the working directory `/data/` before running the notebook.

## How to Set Up and Run the Project

### Notebook: Final_Integrated_Analysis.ipynb
1. Upload the notebook to Jupyter notebook.
2. Ensure all CSV datasets used in the project are present in a `/data/` folder.
3. Install the required libraries using pip if not already installed:
```python
!pip install pandas numpy matplotlib seaborn scikit-learn xgboost nltk folium
```
4. Run the notebook cells in sequence.

## Folder Structure
/project/
Final_Integrated_Analysis.ipynb
/data/
   pavement_info.csv
   traffic.csv
   weather.csv
   customer_reviews.csv
/outputs/
     model_metrics
     cluster_visualization
     README.txt

## Tasks Breakdown

### 1. Data Loading
- Loaded all input CSV datasets using pandas.

### 2. Data Cleaning and Preprocessing
- Removed NA, standardized columns, merged datasets.

### 3. Feature Engineering
- Created new features such as traffic_pressure, normalized_temp, etc.

### 4. Modeling
- Performed regression: Linear Regression, Decision Tree, Random Forest.
- Classification: Naive Bayes, SVM.
- Clustering: K-Means.
- Anomaly Detection: Isolation Forest.
- Topic Modeling: LDA.

### 5. Evaluation
- Used R², MSE, Accuracy, Precision, Recall, F1-score for evaluation.

## Results Snapshot
| Model                  | Accuracy / Score |
|-----------------------|------------------|
| Linear Regression      | R² = 0.79        |
| Decision Tree          | R² = 0.66        |
| Random Forest          | R² = 0.89        |
| Naive Bayes (Text)     | Acc = 74.67%     |
| SVM (Text)             | Acc = 93.33%     |
| Isolation Forest       | Anomalies Detected |
| K-Means Clustering     | Grouped areas into Good/Moderate/Bad |
| LDA                    | 3 Topics extracted from pavement reviews |

## NLP & Text Analysis
NLP & Text Classification
Customer Reviews: Preprocessed textual reviews using standard NLP steps like tokenization, lowercasing, stopword removal, and noise addition.

## Modeling Techniques:
Naive Bayes Classifier: Applied for sentiment classification with good baseline performance.
Support Vector Machine (SVM): Achieved high classification accuracy and precision in detecting positive/negative sentiments.
Evaluation Metrics: Reported Accuracy, Precision, Recall, and F1-Score for both classifiers.

## Topic Modeling (LDA)
Applied Latent Dirichlet Allocation (LDA) to uncover key themes in pavement-related feedback.
Topics Identified:
Cracking and pothole issues
Heavy traffic impact
Overlay repair and maintenance quality

## Model Comparison
Conducted a comprehensive comparison across all models used in the project:
Regression Models: Linear, Decision Tree, Random Forest
Classification Models: Naive Bayes, SVM
Evaluated using:
Regression Metrics: R² Score, Mean Squared Error
Classification Metrics: Accuracy, Precision, Recall, F1-Score



## Notes
- Please ensure column alignment during dummy encoding.
- GBM and RF models need feature-matched dummy datasets.
- Text models need review data with sentiment labels.

## Troubleshooting
- Restart runtime on kernel crash.
- Ensure all dependent libraries are installed.
- Match train/test columns during predictions.
