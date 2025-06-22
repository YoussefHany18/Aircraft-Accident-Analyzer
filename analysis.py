import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('FDR_dataset.csv')


label_map = {
    'Stall': 0,
    'Engine Failure': 1,
    'Pilot Error': 2,
    'Loss of Control': 3,
    'Hard Landing': 4,
    'Runway Overrun': 5,
    'Mid-Air Collision Risk': 6,
    'Fire/Smoke Onboard': 7,
    'Uncommanded Pitch Event': 8,
    'Landing Gear Failure': 9
}
df['AccidentTypeEncoded'] = df['AccidentType'].map(label_map)




# Split to inputs and output
x = df.drop(['AccidentType', 'AccidentTypeEncoded'], axis=1)
y_encoded = df['AccidentTypeEncoded'].astype(int)  

#splitting 80% 20% train / test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)


# Train model
rf_model = RandomForestClassifier(n_estimators=1500, random_state=42)
rf_model.fit(x_train, y_train)


# Evaluation
from sklearn.metrics import accuracy_score

y_pred = rf_model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("Model Accuracy:", round(accuracy * 100,2),"%")



#---------------------------------------------------

# Save the trained model

import joblib
joblib.dump(rf_model, "model.pkl", compress=3)


joblib.dump(label_map, 'label_map.pkl')



