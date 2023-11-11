import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Download stopwords from NLTK
nltk.download('stopwords')

# Load the true and fake news datasets
true_data = pd.read_csv(r'C:\Users\jagan\Downloads\archive\True.csv')
fake_data = pd.read_csv(r'C:\Users\jagan\Downloads\archive\Fake.csv') 

# Label the data as 'true' and 'fake' based on the source dataset
true_data['label'] = 'true'
fake_data['label'] = 'fake'

# Concatenate the datasets
data = pd.concat([true_data, fake_data])

# Convert text to lowercase
data['text'] = data['text'].str.lower()

# Remove punctuation from the text using regular expressions
data['text'] = data['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Remove stopwords from the text using NLTK stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Split the data into features (X) and labels (y) for training and testing
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a TF-IDF vectorizer with a maximum of 5000 features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Transform the training and test data into TF-IDF vectors
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create a Logistic Regression model and train it on the TF-IDF transformed training data
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_tfidf)

# Calculate and print various performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='fake')
recall = recall_score(y_test, y_pred, pos_label='fake')
f1 = f1_score(y_test, y_pred, pos_label='fake')
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_tfidf)[:, 1])

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")
# Output confusion matrix in console
print("Confusion Matrix:")
print(cm)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["True", "Fake"], yticklabels=["True", "Fake"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Print evaluation metrics

