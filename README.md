
# Heart-Disease-Prediction-Using-Random-Forest

📜 **Description**

This project aims to predict whether a person has heart disease based on various health parameters using Machine Learning (Random Forest Classifier). The model is trained and saved as `model.pkl` and can make predictions on new data provided in the `heart.csv` file.

### Heart Disease Prediction Dataset

The dataset is used for predicting heart disease based on various medical attributes of individuals. It contains information about several factors related to heart health, such as age, sex, blood pressure, cholesterol levels, and other cardiovascular risk indicators. The goal is to predict whether an individual has heart disease or not, based on these factors.

### Dataset Columns

- **Age**: Age of the individual (in years).
- **Sex**: Sex of the individual (1 = male, 0 = female).
- **cp (Chest Pain Type)**: Type of chest pain experienced by the individual (1-4, with each number representing a different type of chest pain).
  - 1: Typical angina
  - 2: Atypical angina
  - 3: Non-anginal pain
  - 4: Asymptomatic
- **trestbps (Resting Blood Pressure)**: The resting blood pressure of the individual (in mm Hg).
- **chol (Cholesterol)**: Serum cholesterol levels (in mg/dl).
- **fbs (Fasting Blood Sugar)**: Whether the individual's fasting blood sugar is greater than 120 mg/dl (1 = true, 0 = false).
- **restecg (Resting Electrocardiographic Results)**: The results of the electrocardiogram during rest (0, 1, or 2, representing different types of results).
  - 0: Normal
  - 1: Having ST-T wave abnormality (possibly due to ischemia)
  - 2: Showing probable or definite left ventricular hypertrophy
- **thalach (Maximum Heart Rate Achieved)**: The maximum heart rate achieved during exercise (in beats per minute).
- **exang (Exercise Induced Angina)**: Whether the individual experienced angina (chest pain) during exercise (1 = yes, 0 = no).
- **oldpeak**: Depression induced by exercise relative to rest (measured in ST depression).
- **slope**: Slope of the peak exercise ST segment (1-3, with each number representing a different type of slope).
  - 1: Upsloping
  - 2: Flat
  - 3: Downsloping
- **ca (Number of Major Vessels Colored by Fluoroscopy)**: Number of major vessels (0-3) colored by fluoroscopy.
- **thal (Thalassemia)**: Thalassemia diagnosis (1, 2, 3, or 0 representing different conditions).
  - 1: Normal
  - 2: Fixed defect
  - 3: Reversible defect
- **Prediction**: The target variable indicating the prediction result (whether the person has heart disease or not).
  - "The person has heart disease" or "The person does not have heart disease".

### Purpose of the Dataset

The purpose of this dataset is to help healthcare professionals and researchers predict the likelihood of heart disease in individuals based on medical attributes. This data can be used in machine learning models for classification tasks, such as decision trees, logistic regression, or neural networks, to predict the risk of heart disease.

---

⚙️ **Requirements**

- Python 3.x
- Pandas
- NumPy
- Scikit-Learn
- Joblib

💡 **Installation Steps**

1️⃣ Clone the repository:
```bash
git clone https://github.com/your-repo/heart_disease_prediction.git
```

2️⃣ Navigate to the project folder:
```bash
cd heart_disease_prediction
```

3️⃣ Install the required libraries:
```bash
pip install -r requirements.txt
```

🧐 **How to Run the Project**

1️⃣ Run the `main.py` file:
```bash
python main.py
```

2️⃣ The output will be saved in the `results/predictions.csv` file.

🛠️ **Model Training Script**

The model is a Random Forest Classifier trained on the `heart.csv` dataset. The `model.pkl` file contains both the `StandardScaler` and the trained model.

📌 **Sample Output in `results/predictions.csv`:**

| Age | Sex | CP | Resting BP | Chol | FBS | Rest ECG | Max HR | Exang | Oldpeak | Slope | CA | Thal | Prediction                   |
|-----|-----|----|------------|------|-----|----------|--------|--------|---------|-------|----|------|----------------------------|
| 63  | 1   | 3  | 145        | 233  | 1   | 0        | 150    | 0      | 2.3     | 0     | 0  | 1    | The person has heart disease |
| 37  | 1   | 2  | 130        | 250  | 0   | 1        | 187    | 0      | 3.5     | 0     | 0  | 2    | The person has heart disease |
| 41  | 0   | 1  | 130        | 204  | 0   | 0        | 172    | 0      | 1.4     | 2     | 0  | 2    | The person has heart disease |
| 56  | 1   | 2  | 120        | 236  | 0   | 1        | 178    | 0      | 0.8     | 2     | 0  | 2    | The person has heart disease |
| 57  | 0   | 0  | 140        | 241  | 0   | 1        | 123    | 1      | 0.2     | 1     | 0  | 3    | The person does not have heart disease |
| 48  | 1   | 3  | 138        | 275  | 0   | 0        | 182    | 0      | 0.0     | 2     | 0  | 2    | The person has heart disease |
| 54  | 1   | 1  | 150        | 195  | 0   | 1        | 150    | 0      | 1.0     | 2     | 0  | 3    | The person does not have heart disease |
| 62  | 0   | 2  | 160        | 164  | 0   | 0        | 145    | 0      | 6.2     | 3     | 3  | 3    | The person has heart disease |
| 43  | 1   | 0  | 120        | 177  | 0   | 1        | 120    | 1      | 2.5     | 1     | 0  | 2    | The person does not have heart disease |
| 50  | 0   | 2  | 140        | 217  | 0   | 1        | 169    | 0      | 0.0     | 2     | 0  | 2    | The person has heart disease |

🎯 **Future Improvements**

- Deploy the model using Flask or FastAPI
- Integrate a web-based UI for user interaction
- Real-time data visualization
