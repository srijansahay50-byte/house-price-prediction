# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Step 2: Load dataset (your correct path)
data = pd.read_csv('C:/Users/srija/OneDrive/Desktop/dataset/archive (2)/house_prices.csv')

print("Dataset Preview:")
print(data.head())

print("\nColumns in dataset:")
print(data.columns)

# Step 3: Select features and target
# Using better features for higher accuracy
X = data[['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'floors']]
y = data['price']

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train model (better than Linear Regression)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train model (better than Linear Regression)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Step 8: User input
print("\n--- House Price Prediction ---")

bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = float(input("Enter number of bathrooms: "))
sqft = float(input("Enter area (sqft_living): "))
grade = int(input("Enter house grade (1-13): "))
floors = float(input("Enter number of floors: "))

# Step 9: Predict price
predicted_price = model.predict([[bedrooms, bathrooms, sqft, grade, floors]])

print(f"\n🏠 Estimated House Price: ₹{predicted_price[0]:,.2f}")
