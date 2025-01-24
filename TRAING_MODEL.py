# importing sources
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
# loading data
file_path = "C:\\Users\\harib\\Downloads\\Synthetic_Farmer_Advisory_Dataset.xlsx"  
df = pd.read_excel(file_path)
# Separating Numerical data and text data
for column in df.columns:
    if df[column].dtype in ['float64', 'int64']:  
        df[column] = df[column].fillna(df[column].mean())
    else:  
        df[column] = df[column].fillna(df[column].mode()[0])

#labels of dataset Required
label_encoders = {
    "Crop Stage": LabelEncoder(),
    "Any Cat Event": LabelEncoder(),
    "Agro Advisory": LabelEncoder()
}


for column, encoder in label_encoders.items():
    df[column] = encoder.fit_transform(df[column])


X = df[["Crop Stage", "Any Cat Event"]]  
y = df["Agro Advisory"]  

#Training the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
# storing the trained data with pickle
output_path = "farmer_advisory_model.pkl"  
with open(output_path, 'wb') as f:
    pickle.dump((model, label_encoders), f)

print(f"Model and encoders saved to {output_path}")
#Predicting advice based on  categories
def get_row_wise_predictions(df, model, label_encoders):
    X_input = df[["Crop Stage", "Any Cat Event"]].values

    predictions = model.predict(X_input)
    advisories = label_encoders['Agro Advisory'].inverse_transform(predictions)
    df['Predicted Agro Advisory'] = advisories
    return df
#call
df_with_predictions = get_row_wise_predictions(df, model, label_encoders)

df_with_predictions.to_csv("predictions_with_agro_advisory.csv", index=False)
print("Predictions saved to predictions_with_agro_advisory.csv")

df_with_predictions.to_json("predictions_with_agro_advisory.json", orient="records")
print("Predictions saved to predictions_with_agro_advisory.json")
