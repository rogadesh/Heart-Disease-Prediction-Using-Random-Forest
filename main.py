from utils.helper_functions import predict_heart_disease_for_csv

csv_path = "data/heart.csv"
output_file = predict_heart_disease_for_csv(csv_path)
print(f"Predictions saved to: {output_file}")
