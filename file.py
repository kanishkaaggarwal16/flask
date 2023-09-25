# %%


# %% [markdown]
# Kanishka_Aggarwal_40821058_Practical-1

# %% [markdown]
# # PRACTICAL-1

# %% [markdown]
# Data Preprocessing

# %%
import pandas as pd
import matplotlib.pyplot as plt
#importing data
data = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\mdcm\Investments_VC.csv", encoding="unicode_escape")

# %%
#knowing data
data.head(5)

# %%
data.isnull().sum()  #checking for nulls

# %%
data.info()

# %%
#replacing whitespaces
data.columns = data.columns.str.replace(' ', '_')
data.columns

# %%
data['city'] = data['city'].str.lower() #basic preprocessing

# %%
#finding unique values for our target column
data['status'].unique

# %%
#then counting them
counts = data['status'].value_counts() 
counts

# %%
#after counting making a bar plot for it
plt.figure(figsize=(6, 3))
counts.plot(kind='bar', color='magenta')
plt.title('Value Counts of "status"')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# %%
#renaming the columns for easy feature namw retrieval
data.rename(columns={'_funding_total_usd_': 'total_f'}, inplace=True)
data.rename(columns={'_market_': 'market'}, inplace=True)

# %%
#converting object to numeric 
data['total_f'] = pd.to_numeric(data['total_f'].str.replace(',', ''), errors='coerce')
data['total_f'].fillna(0, inplace=True)
# Fill NaN values with 0 for all numeric columns
numeric_columns = data.select_dtypes(include=['float64']).columns
data[numeric_columns] = data[numeric_columns].fillna(0)

# %%
# Using overall mode for mode imputation
mode_value = data['market'].mode()[0]
data['market'].fillna(mode_value, inplace=True)

# %%
# Data exploration
unique_countries = data['country_code'].unique()
unique_states = data['state_code'].unique()

# %%
#encoding specific market column
freq_encoding = data['market'].value_counts(normalize=True)
data['market_encoded'] = data['market'].map(freq_encoding)
data['market_encoded'] 

# %%
#data['region'].fillna('Unknown', inplace=True)

# %%
country_counts = data.groupby('country_code').size()

# %%
data.fillna('Unknown', inplace=True)

# %%
#Label Encoding (for columns with ordinal relationships:
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['country_code_encoded'] = label_encoder.fit_transform(data['country_code'])
data['state_code_encoded'] = label_encoder.fit_transform(data['state_code'])

# %%
# Replace NaN with the most frequent category
most_frequent_category = data['status'].mode()[0]
data['status'] = data['status'].replace("Unknown", most_frequent_category)

# %%
data['status']

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# Kanishka_Aggarwal_40821058_Practical-2

# %% [markdown]
# # PRACTICAL-2

# %% [markdown]
# Appling the Model : A Gradient Boosting Classifier (GBC) model and Random Forest Classifier

# %%
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

# %%
# make a new column named success having target values Y and selected features as X for training
data['success'] = data['status'].apply(lambda status: 1 if status == 'operating' else (2 if status == 'acquired' else 0))

selected_features = [
    'total_f',
    'funding_rounds',
    'seed',
    'venture',
   # 'angel',
    #'private_equity',
    'market_encoded',
   # 'equity_crowdfunding',
   # 'convertible_note',
    'debt_financing',
    'country_code_encoded',
    'state_code_encoded',
]

# %%
#defining X and Y
y = data['success']
X = data[selected_features]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)

# %%
X.info()

# %%


# %% [markdown]
# 

# %%
#correlation matrix
import seaborn as sns
correlation_matrix = X.corr()
plt.figure(figsize=(6, 3))  
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# Gradient Boosting Classifier (GBC) model 

# %%
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train, y_train)
prediction=gb_classifier.predict(X_test)
prediction

# %%
from sklearn.metrics import confusion_matrix

# Evaluate the model
accuracy_gb = accuracy_score(y_test, prediction)
confusion_matrix_gb = confusion_matrix(y_test, prediction)
classification_report_gb = classification_report(y_test, prediction)

# Print evaluation results
print(f"Accuracy: {accuracy_gb}")
print(f"Confusion Matrix:\n{confusion_matrix_gb}")
print(f"Classification Report:\n{classification_report_gb}")

# %% [markdown]
# Random Forest Classifier Model

# %%
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pr=model.predict(X_test)

# %%
from sklearn.metrics import confusion_matrix

# Evaluate the model
accuracy = accuracy_score(y_test, y_pr)
cm = confusion_matrix(y_test, y_pr)
classification_report = classification_report(y_test, y_pr)

# Print evaluation results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{cm}")
print(f"Classification Report:\n{classification_report}")

# %%
feature_importances = model.feature_importances_
#ploting features importance
plt.figure(figsize=(6, 3))
plt.bar(range(len(feature_importances)), feature_importances, tick_label=selected_features,color='green')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importance Analysis')
plt.xticks(rotation=90)
plt.show()

# %%
#comparison between models according to their accuracy
model_names = ['GBC', 'Random Forest']

# Define the corresponding accuracy scores
accuracy_scores = [accuracy_gb, accuracy] 

# Create a bar chart
plt.figure(figsize=(4,3))
plt.bar(model_names, accuracy_scores, color=['blue', 'green'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1) 
plt.show()

# %% [markdown]
# CONCLUSION:
# Trained a Gradient Boosting Classifier (GBC) model using some training data; GBC revealed highest Accuracy then Random Forest which is 0.8.

# %% [markdown]
# Making function for preprocessing and training 

# %%
def preprocess_data(data):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    object_columns = data.select_dtypes(include=['object']).columns

    for column in object_columns:
        data[column] = label_encoder.fit_transform(data[column])

    scaled_features = scaler.fit_transform(data)

    data[data.columns] = scaled_features  

    return data

# %%
def train_model_gbc(X,y):
    from sklearn.ensemble import GradientBoostingClassifier
    gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_classifier.fit(X_train, y_train)
    return gb_classifier

# %% [markdown]
# Then save the trained model using the pickle

# %%
model=train_model_gbc(X_train,y_train)
with open('model.pkl','wb') as model_file:
    pickle.dump(model, model_file)

# %%
with open('preprocess_data.pkl','wb') as model_file:
    pickle.dump(preprocess_data, model_file)

# %%


# %%


# %%


# %%


# %% [markdown]
# Kanishka_Aggarwal_40821058_Practical-3

# %% [markdown]
# # PRACTICAL-3

# %% [markdown]
# Deployment without Pipeline

# %%
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

# %% [markdown]
# creating new Dataframe for Testing Data

# %%
testing_data=testing_data = {
    'total_f': [4000000.0],
    'funding_rounds': [2.0],
    'seed': [0.0],
    'venture': [4000000.0],
    #'angel': [0.0],
    #'private_equity': [0.0],
    'market_encoded': [0.02177],
   # 'equity_crowdfunding': [0.0],
    #'convertible_note': [0.0],
    'debt_financing': [0.0],
    'country_code_encoded': [110],
    'state_code_encoded': [6],
}

testing_df = pd.DataFrame(testing_data)
y_true= np.array([1])

# %%
testing_df

# %% [markdown]
# Loading our Model

# %%
with open('model.pkl','rb') as model_file:
    trained_model=pickle.load(model_file)
with open('preprocess_data.pkl','rb') as model_file:
    preprocess_data=pickle.load(model_file)

# %% [markdown]
# Testing our model with testing data

# %%
test=preprocess_data(testing_df)
y_pred=model.predict(test)
print(y_pred)

# %%
#1 is for Operating

# %% [markdown]
# Evaluation

# %%
# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


def preprocess_data(data):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    object_columns = data.select_dtypes(include=['object']).columns

    for column in object_columns:
        data[column] = label_encoder.fit_transform(data[column])

    scaled_features = scaler.fit_transform(data)

    data[data.columns] = scaled_features  

    return data

def train_model_gbc(X,y):
    from sklearn.ensemble import GradientBoostingClassifier
    gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_classifier.fit(X_train, y_train)
    return gb_classifier

model=train_model_gbc(X_train,y_train)
with open('model.pkl','wb') as model_file:
    pickle.dump(model, model_file)


with open('preprocess_data.pkl','wb') as model_file:
    pickle.dump(preprocess_data, model_file)





