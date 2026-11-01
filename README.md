 🌧️ Rainfall Pattern Analysis and Prediction Using Machine Learning

## 📌 Project Overview

Rainfall Pattern Analysis and Prediction is a Python-based data science project developed as part of my academic coursework at Lovely Professional University.

The project focuses on analysing historical rainfall data across India from **1901 to 2015**, covering 36 meteorological subdivisions. It uses data preprocessing, Exploratory Data Analysis (EDA), data visualization, machine learning and statistical techniques to understand rainfall patterns and estimate future rainfall.
An interactive web dashboard has been developed using **Streamlit**, allowing users to explore rainfall information, compare different regions, evaluate prediction models and identify unusual rainfall observations.

## 🎯 Project Objectives

- Analyse historical rainfall data across different meteorological subdivisions of India.
- Understand monthly, seasonal and annual rainfall patterns.
- Compare average rainfall across different regions.
- Apply multiple regression algorithms for rainfall prediction.
- Evaluate and compare machine learning models using performance metrics.
- Detect unusually high or low rainfall observations using the IQR method.
- Develop an interactive dashboard for easy data exploration and analysis.

## 📂 Dataset Description

The project uses the dataset **Area Weighted Monthly, Seasonal and Annual Rainfall (in mm) for 36 Meteorological Subdivisions**.

| Dataset Information | Details |
|---|---|
| Dataset Period | 1901–2015 |
| Total Records | 4,116 |
| Total Columns | 19 |
| Meteorological Subdivisions | 36 |
| File Format | CSV |
| Main Target Variable | ANNUAL |

The dataset includes:
- Monthly rainfall values (January–December)
- Annual rainfall
- Seasonal rainfall values
- Subdivision names
- Year-wise records

**Dataset Source:** Government of India Open Data Portal  
https://www.data.gov.in/resource/area-weighted-monthly-seasonal-and-annual-rainfall-mm-36-meteorological-subdivisions-1901

## 🛠️ Technologies and Libraries

| Technology | Application |
|---|---|
| Python | Project development |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical calculations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning implementation |
| Streamlit | Interactive dashboard development |

## 🔍 Exploratory Data Analysis (EDA)

The project performs exploratory analysis to understand the structure and characteristics of the rainfall dataset.

The analysis includes:
- Dataset inspection and summary statistics.
- Missing-value identification.
- Duplicate-record checking.
- Data type conversion and cleaning.
- Monthly rainfall analysis.
- Annual rainfall trend analysis.
- Subdivision-wise rainfall comparison.
- Seasonal rainfall contribution analysis.

## 🤖 Machine Learning Models Implemented

Five regression models are implemented to estimate annual rainfall and compare their prediction performance.

1. **Simple Linear Regression:** Studies the relationship between year and annual rainfall.
2. **Polynomial Regression:** Uses a quadratic relationship to represent nonlinear trends.
3. **Multiple Linear Regression:** Uses rainfall values from the previous three years as input features.
4. **Decision Tree Regression:** Uses decision-based splitting to estimate rainfall.
5. **Random Forest Regression:** Combines multiple decision trees to generate predictions.

### Model Evaluation Metrics

The models are evaluated using:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Coefficient of Determination

A chronological 80:20 train-test split is used to maintain the order of historical rainfall records.

## 📊 Interactive Dashboard Features

The Streamlit application contains four main sections.

### 1. Dashboard
- Displays important dataset information through KPI cards.
- Shows annual rainfall trends.
- Presents average rainfall across subdivisions.
- Visualizes monthly rainfall patterns.
- Displays seasonal rainfall contributions.

### 2. Rainfall Explorer
- Allows users to select a meteorological subdivision.
- Provides a year-range filter.
- Displays annual rainfall trends for selected regions.
- Shows average monthly rainfall.
- Provides filtered rainfall records.

### 3. Prediction Lab
- Implements and compares five regression models.
- Displays MAE, MSE, RMSE and R² scores.
- Visualizes model performance through comparison charts.
- Provides next-year rainfall estimates.

### 4. Anomaly Analysis
- Applies the Interquartile Range (IQR) method.
- Calculates lower and upper anomaly limits.
- Displays annual rainfall distribution using a boxplot.
- Highlights unusually high or low rainfall observations.
- Provides anomaly visualization and records.

## 📈 Key Learning Outcomes

Through this project, I gained practical experience in:

- Working with real-world historical datasets.
- Data cleaning and preprocessing using Pandas.
- Performing exploratory data analysis.
- Creating meaningful data visualizations.
- Implementing and comparing regression algorithms.
- Understanding machine learning evaluation metrics.
- Applying statistical techniques for anomaly detection.
- Developing an interactive data science application using Streamlit.

## ⚙️ Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/Anjusihag/-Rainfall-Pattern-Analysis-and-Prediction.git
```

### Step 2: Open the Project Folder

```bash
cd -Rainfall-Pattern-Analysis-and-Prediction
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

Make sure the rainfall CSV file is placed in the same directory as the Python application.

## 📁 Project Structure

```text
Rainfall-Pattern-Analysis-and-Prediction/
│
├── app.py
├── rainfall_area-wt_sd_1901-2015.csv
├── requirements.txt
├── README.md
└── screenshots/
```

## ⚠️ Limitations

- Predictions are based on historical rainfall values.
- The models do not use additional climate variables such as temperature, humidity or atmospheric pressure.
- Predictions are educational estimates and should not be treated as official weather forecasts.
- IQR-based anomalies do not automatically indicate floods or droughts.

## 🚀 Future Scope

- Implement advanced time-series forecasting models such as ARIMA.
- Include additional climate-related variables.
- Add geographical maps for subdivision-wise rainfall visualization.
- Improve model validation using time-series cross-validation.
- Enhance the dashboard with additional interactive analytical features.

## 🎓 Academic Information

**Project Title:** Rainfall Pattern Analysis and Prediction  
**Course:** Predictive Analytics  
**University:** Lovely Professional University  
**School:** School of Computing Science and Engineering

## 👩‍💻 Author

**Anju**  
B.Tech CSE – Data Science  
Lovely Professional University

---

⭐ If you find this project interesting, feel free to explore the repository!
