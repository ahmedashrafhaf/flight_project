#                                                     ✈️ Flight Delay Prediction & Analytics


## 📌 Project Overview

Flight delays are one of the most common operational challenges in the aviation industry. They can affect passengers, airlines, airport operations, crew scheduling, and overall transportation efficiency.

This project develops an **end-to-end Machine Learning solution** capable of predicting whether a flight will arrive more than **15 minutes late**, while also providing an interactive analytics platform for exploring flight delay patterns.

The project covers the complete Data Science lifecycle:

> **Data → Cleaning → EDA → Feature Engineering → Preprocessing → Modeling → Evaluation → Interpretation → Deployment**

The final solution combines a high-performing **HistGradientBoostingClassifier** with an interactive **Streamlit dashboard**.

---

# 🎯 Business Problem

The objective is to answer two main questions:

### 1. Prediction

> **Will this flight arrive more than 15 minutes late?**

The model produces a probability of delay and converts it into a binary prediction using an optimized classification threshold.

### 2. Analytics

> **What patterns and factors are associated with flight delays?**

The analytical dashboard allows users to investigate delay behavior across:

* Airlines
* Airports
* Months
* Days of the week
* Departure hours
* Flight distance
* Departure delay
* Taxi times

---

# 🎯 Machine Learning Objective

The target variable is:

```text
IS_DELAYED = 1  → Arrival delay > 15 minutes
IS_DELAYED = 0  → Arrival delay ≤ 15 minutes
```

This makes the problem a **Binary Classification** task.

---

# 📊 Dataset

The project uses a large-scale flight dataset containing approximately:

* **2,000,000 flight records**
* **32 original features**

After data cleaning:

| Metric           |     Value |
| ---------------- | --------: |
| Original Records | 2,000,000 |
| Clean Records    | 1,942,769 |
| On-Time Flights  | 1,598,894 |
| Delayed Flights  |   343,875 |
| Delay Rate       |    17.70% |

The dataset contains information related to:

* Flight schedules
* Airlines
* Origin airports
* Destination airports
* Flight distance
* Taxi times
* Departure delays
* Arrival delays

> **Note:** The complete dataset is not included in this repository because of its large size.

---

# 🔄 Complete Data Science Workflow

## 1. Data Collection

The project starts with a large-scale flight dataset containing millions of observations.

The raw dataset contains operational and scheduling information that can be used to investigate and model flight delays.

---

# 2. Data Understanding

The dataset was initially inspected to understand:

* Dataset dimensions
* Feature types
* Missing values
* Duplicate records
* Numerical distributions
* Categorical variables
* Target distribution
* Potential data quality problems

Initial dataset:

```text
Rows:    2,000,000
Columns: 32
```

---

# 3. Data Cleaning

Several preprocessing steps were performed before modeling.

### Removed invalid flight records

Cancelled and diverted flights were excluded because they do not represent normal arrival-delay outcomes.

### Handled missing values

Records with missing values in required modeling fields were removed or handled during preprocessing.

### Created the target

```python
IS_DELAYED = (ARR_DELAY > 15).astype(int)
```

After cleaning:

```text
1,942,769 records
```

---

# 4. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure of the data and discover relationships with flight delays.

### Key areas investigated

#### Delay Distribution

Analyzed the distribution of delayed versus non-delayed flights.

#### Airline Analysis

Compared delay behavior across airlines.

#### Airport Analysis

Investigated delay patterns across origin and destination airports.

#### Time Analysis

Analyzed delays by:

* Month
* Day of week
* Departure hour
* Arrival hour

#### Flight Distance

Investigated the relationship between flight distance and delays.

#### Departure vs Arrival Delay

Examined how departure delays relate to arrival delays.

EDA helped identify meaningful variables for the Machine Learning stage.

---

# 5. Target Distribution & Class Imbalance

The target variable is imbalanced:

| Class   |   Records | Percentage |
| ------- | --------: | ---------: |
| On Time | 1,598,894 |     82.30% |
| Delayed |   343,875 |     17.70% |

Because the majority class represents more than 80% of observations, **accuracy alone is not sufficient** for evaluating the models.

Therefore, the project also focuses on:

* Precision
* Recall
* F1-Score
* ROC-AUC
* PR-AUC
* Confusion Matrix

---

# 6. Train / Validation / Test Split

The dataset was split using **stratified sampling** to preserve the target distribution.

The modeling workflow uses:

```text
64% Training
16% Validation
20% Testing
```

The validation set is used for model/threshold decisions, while the test set is kept untouched for final evaluation.

This helps reduce evaluation bias and prevents test-set information from influencing threshold selection.

---

# 7. Feature Selection

The final modeling features include:

```text
MONTH
DAY_OF_WEEK
AIRLINE_CODE
ORIGIN
DEST
DISTANCE
TAXI_OUT
TAXI_IN
CRS_DEP_TIME
CRS_ARR_TIME
DEP_DELAY
```

These variables represent flight scheduling, airline, route, distance, taxi operations, and departure-delay information.

---

# 8. Feature Engineering

Feature engineering was applied to transform raw variables into more useful representations.

## Departure Hour

Scheduled departure time was converted into an hour-based feature:

```python
DEP_HOUR = CRS_DEP_TIME // 100
```

## Arrival Hour

```python
ARR_HOUR = CRS_ARR_TIME // 100
```

The original scheduled time columns were then removed.

---

# 9. Cyclical Time Encoding

Time is cyclical.

For example:

```text
23:00 → 00:00
```

are close in real-world time even though their numerical values are far apart.

To represent this relationship, sine and cosine transformations were applied.

### Departure Time

```text
DEP_HOUR_SIN
DEP_HOUR_COS
```

### Arrival Time

```text
ARR_HOUR_SIN
ARR_HOUR_COS
```

This allows the model to capture the cyclical nature of time.

---

# 10. Categorical Encoding

Categorical airport features were transformed using **target encoding**.

### Origin

```text
ORIGIN → ORIGIN_TE
```

### Destination

```text
DEST → DEST_TE
```

A smoothing factor of:

```text
20
```

was used.

The mappings were fitted **only on the training data** and then applied to validation and test data.

This prevents information leakage.

---

# 11. Airline Encoding

`AIRLINE_CODE` was transformed using one-hot encoding.

The resulting categorical variables were aligned across:

```text
Train
Validation
Test
```

to ensure identical feature structures.

---

# 12. Outlier Handling

Extreme values were capped using the **99th percentile calculated from the training data**.

Applied to:

| Feature   | 99th Percentile Cap |
| --------- | ------------------: |
| DISTANCE  |                2588 |
| TAXI_OUT  |                  52 |
| TAXI_IN   |                  33 |
| DEP_DELAY |                 191 |

The limits were learned from training data and then applied consistently to validation and test sets.

---

# 13. Feature Scaling

**RobustScaler** was selected because it is less sensitive to extreme values than standard scaling.

The following features were scaled:

```text
DISTANCE
TAXI_OUT
TAXI_IN
DEP_DELAY
ORIGIN_TE
DEST_TE
```

The scaler was fitted exclusively on the training set.

---

# 14. Final Preprocessed Feature Set

After feature engineering and preprocessing, the final feature representation contains:

* Numerical features
* Engineered time features
* Target-encoded airport features
* One-hot encoded airline features
* Robust-scaled continuous variables

All train, validation, and test datasets were aligned to the same feature structure.

---

# 🤖 15. Model Development

Multiple classification algorithms were evaluated.

### Models

1. **Logistic Regression**
2. **HistGradientBoostingClassifier**

The models were compared using multiple evaluation metrics.

---

# 🧪 16. Model Evaluation

Because the dataset is imbalanced, model evaluation was not based on accuracy alone.

The following metrics were used:

### Accuracy

Overall percentage of correct predictions.

### Precision

Percentage of predicted delayed flights that were actually delayed.

### Recall

Percentage of actual delayed flights correctly detected.

### F1-Score

Harmonic mean of Precision and Recall.

### ROC-AUC

Measures the model's ability to distinguish between delayed and non-delayed flights.

### PR-AUC

Particularly useful for evaluating performance when the positive class is relatively less frequent.

---

# 🏆 17. Model Comparison

| Model                |   Accuracy |  Precision |     Recall |   F1-Score |    ROC-AUC |     PR-AUC |
| -------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| HistGradientBoosting | **95.65%** | **91.35%** |     83.29% |     87.14% | **98.07%** | **94.66%** |
| Final HGB @ 0.45     |     95.61% |     89.88% | **84.75%** | **87.24%** |     98.06% |     94.64% |
| Logistic Regression  |     93.22% |     75.25% | **91.91%** |     82.75% |     97.66% |     93.60% |

---

# 🥇 18. Final Model

The final model is based on:

```text
HistGradientBoostingClassifier
```

The classification threshold was adjusted from:

```text
0.50 → 0.45
```

The lower threshold increases the model's ability to identify delayed flights.

### Final Test Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **95.61%** |
| Precision | **89.88%** |
| Recall    | **84.75%** |
| F1-Score  | **87.24%** |
| ROC-AUC   | **98.06%** |
| PR-AUC    | **94.64%** |

The threshold of `0.45` provides a slightly better balance between precision and recall for the project's objective.

---

# 🎯 19. Threshold Optimization

Instead of automatically using:

```text
threshold = 0.50
```

different thresholds were evaluated.

The threshold determines when a predicted probability becomes a positive delay prediction.

For example:

```text
Probability >= 0.45 → Delayed
Probability < 0.45  → Not Delayed
```

A threshold of **0.45** was selected based on validation performance, with the goal of improving delay detection while maintaining strong precision.

---

# 🔍 20. Model Interpretation

Model behavior can be investigated through:

* Feature importance
* Prediction probabilities
* Error analysis
* Threshold analysis
* Delay patterns across operational variables

The project focuses on understanding **which variables are associated with the model's predictions**, rather than treating the model as a black box.

---

# 📊 21. Streamlit Application

The trained model was integrated into an interactive Streamlit application.

The application is designed as an aviation analytics and prediction platform rather than a simple prediction form.

---

## 🖥️ Application Pages

### ✈️ Flight Prediction

Users can enter flight information and receive:

* Delay probability
* Predicted class
* Risk level
* Flight summary
* Threshold information
* What-if scenario analysis

---

### 📊 Flight Analytics

Interactive analytics include:

* Total flights
* Delayed flights
* On-time flights
* Overall delay rate
* Average departure delay
* Average flight distance

Users can explore delay patterns by:

* Airline
* Month
* Origin
* Destination
* Departure hour
* Flight distance
* Departure delay

---

### 🧠 Model Insights

The application provides model-related insights including:

* Feature importance
* Important operational variables
* Delay patterns
* Airline-level insights
* Airport-level insights
* Time-based patterns

---

### 📈 Model Performance

The application presents:

* Model comparison
* Confusion matrix
* ROC curve
* Precision-Recall curve
* Threshold analysis
* Classification metrics

---

### ℹ️ About the Project

Provides an overview of:

* Business problem
* Dataset
* Data Science workflow
* Machine Learning approach
* Technologies
* Model performance
* Project limitations

---

# ⚡ 22. Performance Optimization

Because the project works with approximately **2 million records**, performance optimization is important.

The Streamlit application uses:

* Data caching
* Model caching
* Efficient Pandas operations
* Precomputed aggregations where appropriate
* Lazy loading where possible

This prevents expensive operations from being repeated unnecessarily during every Streamlit interaction.

---

# 🗂️ 23. Project Structure

```text
flight-delay-prediction/
│
├── app.py
│
├── pages/
│   ├── prediction.py
│   ├── analytics.py
│   ├── model_insights.py
│   ├── model_performance.py
│   └── about.py
│
├── src/
│   ├── preprocessing.py
│   ├── prediction.py
│   ├── analytics.py
│   └── visualization.py
│
├── models/
│   ├── flight_delay_model.pkl
│   ├── scaler.pkl
│   ├── origin_mapping.pkl
│   ├── dest_mapping.pkl
│   └── threshold.pkl
│
├── assets/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠️ 24. Technology Stack

| Category          | Technologies                   |
| ----------------- | ------------------------------ |
| Programming       | Python                         |
| Data Manipulation | Pandas, NumPy                  |
| Visualization     | Matplotlib, Seaborn, Plotly    |
| Machine Learning  | Scikit-learn                   |
| Model             | HistGradientBoostingClassifier |
| Web Application   | Streamlit                      |
| Development       | Jupyter Notebook, VS Code      |
| Version Control   | Git, GitHub                    |

---

# ⚙️ 25. Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/flight-delay-prediction.git
```

Navigate to the project:

```bash
cd flight-delay-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ 26. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will then be available through the local Streamlit server.

---

# 🔐 27. Data & Model Considerations

The complete dataset is intentionally excluded from the GitHub repository because of its size.

Large files, temporary files, virtual environments, credentials, and unnecessary generated files should be excluded through `.gitignore`.

The repository should contain the code and required lightweight model/preprocessing artifacts needed to reproduce or run the application.

---

# ⚠️ 28. Important Modeling Limitation

The current model includes:

```text
DEP_DELAY
TAXI_OUT
TAXI_IN
```

These variables are available after the flight has started.

Therefore, the current system should be interpreted as:

> **Post-Departure Arrival Delay Prediction**

rather than a pure pre-departure prediction system.

A true pre-departure prediction model would require retraining without post-departure variables.

---

# 💡 29. Key Project Outcomes

This project demonstrates the ability to:

* Work with a large-scale dataset
* Perform structured data cleaning
* Conduct exploratory data analysis
* Handle class imbalance
* Engineer meaningful features
* Encode categorical variables
* Handle outliers
* Scale numerical features
* Prevent preprocessing leakage
* Train multiple ML models
* Compare classification models
* Optimize classification thresholds
* Evaluate models using appropriate metrics
* Interpret model behavior
* Build an interactive analytics application
* Deploy a Machine Learning model using Streamlit
* Structure a Data Science project for production-style usage

---

# 🚀 30. Future Improvements

Potential future improvements include:

* Pre-departure delay prediction
* Time-based validation to simulate future flights
* Hyperparameter optimization
* Model calibration
* Advanced explainability using SHAP
* Real-time flight data integration
* Weather data integration
* Airport congestion features
* Aircraft/route-level historical features
* Model monitoring
* Automated retraining pipeline
* Cloud deployment

---

# 📌 31. Disclaimer

The model predictions represent statistical estimates based on historical flight data.

A prediction of delay does not guarantee that a flight will actually be delayed.

The system should be considered a **decision-support and analytical tool**, not a replacement for operational aviation systems.


---

<p align="center">
  <b>✈️ From Raw Flight Data to Machine Learning Insights</b>
</p>

<p align="center">
  Built as an end-to-end Data Science & Machine Learning project.
</p>

