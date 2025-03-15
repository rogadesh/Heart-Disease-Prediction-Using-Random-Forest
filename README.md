# Heart-Disease-Prediction-Using-Random-Forest

📜 Description
This project aims to predict whether a person has heart disease based on various health parameters using Machine Learning (Random Forest Classifier). The model is trained and saved as model.pkl and can make predictions on new data provided in the heart.csv file.

⚙️ Requirements
Python 3.x
Pandas
NumPy
Scikit-Learn
Joblib

💡 Installation Steps

1️⃣ Clone the repository:
bash
Copy code
git clone https://github.com/your-repo/heart_disease_prediction.git

2️⃣ Navigate to the project folder:
bash
Copy code
cd heart_disease_prediction

3️⃣ Install the required libraries:
bash
Copy code
pip install -r requirements.txt

🧐 How to Run the Project

1️⃣ Run the main.py file:
bash
Copy code
python main.py

2️⃣ The output will be saved in the results/predictions.csv file.
🛠️ Model Training Script
The model is a Random Forest Classifier trained on the heart.csv dataset. The model.pkl file contains both the StandardScaler and the trained model.

📌 Sample Output in results/predictions.csv:
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

🎯 Future Improvements
Deploy the model using Flask or FastAPI
Integrate a web-based UI for user interaction
Real-time data visualization