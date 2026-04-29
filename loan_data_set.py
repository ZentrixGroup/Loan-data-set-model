import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load the dataset
df = pd.read_csv('loan_data_set.csv')

# 2. Data Cleaning
# Remove unnecessary columns that don't contribute to the model
df.drop('Loan_ID', axis=1, inplace=True)

# Fill missing values in categorical columns with the most frequent value (mode)
categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill missing values in the numerical column with the median
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

# 3. Label Encoding
# Convert categorical text data into numerical values using factorization
# This ensures consistency for the model and deployment (e.g., Streamlit)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.factorize(df[col])[0]

# 4. Model Building
# Define features (X) and target variable (y)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# 5. Model Export
# Save the trained model as a pickle file for later use in production
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)

