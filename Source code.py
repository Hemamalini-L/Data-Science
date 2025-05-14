# fake_news_pipeline.py

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, ConfusionMatrixDisplay

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load the dataset
data = pd.read_csv('fake_news_detection_dataset.csv')
print('Original dataset shape:', data.shape)

# Remove missing values and duplicates
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
print('After cleaning shape:', data.shape)

# Apply preprocessing
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Save cleaned dataset
data.to_csv('cleaned_fake_news_dataset.csv', index=False)
print("✅ Cleaned dataset saved as 'cleaned_fake_news_dataset.csv'")

# Plot label distribution
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=data, palette='Set2')
plt.title('Distribution of Fake vs Real News')
plt.show()

# Word Clouds
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
fake_news = data[data['label'] == 'Fake']
wordcloud_fake = WordCloud(width=400, height=300, background_color='black').generate(' '.join(fake_news['cleaned_text']))
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis('off')
plt.title('Fake News Word Cloud')

plt.subplot(1, 2, 2)
real_news = data[data['label'] == 'Real']
wordcloud_real = WordCloud(width=400, height=300, background_color='black').generate(' '.join(real_news['cleaned_text']))
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.axis('off')
plt.title('Real News Word Cloud')
plt.show()

# Feature Extraction
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(data['cleaned_text'])
y = data['label']

# Save vectorizer and features
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))
pickle.dump(X, open('tfidf_features.pkl', 'wb'))
pickle.dump(y, open('labels.pkl', 'wb'))
print("✅ TF-IDF features and labels saved")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save trained model
pickle.dump(model, open('best_model.pkl', 'wb'))
print("✅ Model trained and saved as 'best_model.pkl'")

# Evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
y_proba = model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test.map({'Fake': 0, 'Real': 1}), y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Save model and vectorizer to 'model/' directory
os.makedirs('model', exist_ok=True)
with open('model/fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("✅ Model and vectorizer saved in the 'model/' directory.")
