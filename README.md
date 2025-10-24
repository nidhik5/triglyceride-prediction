🧠 Triglyceride Level Prediction Using Machine Learning

A predictive machine learning model that estimates triglyceride levels based on demographic, physiological, and lifestyle data from 22,401 individuals.
This project demonstrates a complete data science workflow — from preprocessing and feature engineering to model training, hyperparameter tuning, and evaluation.

📊 Project Overview

Cardiovascular health depends on early detection of abnormal lipid levels. This project builds a regression-based model to predict triglyceride concentration using clinical and lifestyle variables such as:

Age, gender, and BMI

Blood pressure and cholesterol levels

Liver enzyme concentrations (ALT, AST, GGT)

Lifestyle factors (smoking, alcohol consumption, exercise)

🚀 Key Features

✅ Cleaned and preprocessed real-world medical dataset (22K+ samples)
✅ Built a Linear Regression model optimized via Gradient Descent
✅ Applied ColumnTransformer for efficient scaling and encoding
✅ Used GridSearchCV for hyperparameter tuning to prevent overfitting
✅ Achieved 94% model accuracy on validation data
✅ Generated visual insights with Matplotlib and Seaborn

🧩 Tech Stack

Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Techniques: Linear Regression, Feature Engineering, Hyperparameter Tuning, Data Normalization

🧮 Model Pipeline

Data Cleaning: Removed null values, handled outliers, and standardized units

Feature Engineering:

Scaled continuous variables

Encoded categorical attributes

Model Training: Linear Regression with Gradient Descent optimization

Evaluation: R² score, RMSE, and accuracy metrics

Visualization: Correlation heatmaps and regression plots

📈 Results

Final model achieved 94% accuracy

Low mean absolute error (MAE) indicating high predictive precision

Insights revealed cholesterol levels and liver enzymes as top predictive features

🧠 Learnings

Translating clinical data into actionable ML models

Handling multi-feature health data pipelines

Fine-tuning models using GridSearchCV for reproducibility and generalization

📂 Repository Structure
├── data/
│   └── triglyceride_dataset.csv
├── notebooks/
│   └── Triglyceride_Prediction.ipynb
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate.py
├── results/
│   └── model_performance.png
└── README.md

🧑‍💻 Author

Nidhi Kadam
Machine Learning Engineer | University of Chicago (MS in Applied Data Science, Fall 2025)
🔗 GitHub
 • LinkedIn
