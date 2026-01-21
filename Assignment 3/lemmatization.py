# Assignment 3: Text Cleaning, Lemmatization, Stopword Removal, Label Encoding, TF-IDF

import sys
import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Redirect all prints to output.txt in Assignment 3 folder
sys.stdout = open("Assignment 3/output.txt", "w", encoding="utf-8")

# Download required NLTK resources silently
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

# -------------------------------
# 1. Load Dataset (train.jsonl)
# -------------------------------
data_path = "data/train.jsonl"

records = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

print("\n--- Dataset Loaded ---")
print(df.head())

# Use the 'messages' field as text
texts = df['messages'].dropna().astype(str)

# -------------------------------
# 2. Text Cleaning
# -------------------------------
def clean_text(text):
    text = text.lower()                          # lowercase
    text = re.sub(r'[^a-z\s]', '', text)         # remove punctuation/numbers
    return text

cleaned_texts = texts.apply(clean_text)
print("\n--- Cleaned Text Sample ---")
print(cleaned_texts.head())

# -------------------------------
# 3. Lemmatization
# -------------------------------
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = nltk.word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmas)

lemmatized_texts = cleaned_texts.apply(lemmatize_text)
print("\n--- Lemmatized Text Sample ---")
print(lemmatized_texts.head())

# -------------------------------
# 4. Stopword Removal
# -------------------------------
stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

final_texts = lemmatized_texts.apply(remove_stopwords)
print("\n--- Stopword Removed Text Sample ---")
print(final_texts.head())

# -------------------------------
# 5. Label Encoding
# -------------------------------
# Example: encode sender_labels
label_column = 'sender_labels'
labels = df[label_column].dropna().astype(str)

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

print("\n--- Label Encoding ---")
print("Classes:", encoder.classes_)
print("Encoded Labels Sample:", encoded_labels[:10])

# -------------------------------
# 6. TF-IDF Representations
# -------------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(final_texts)

print("\n--- TF-IDF ---")
print("Vocabulary size:", len(tfidf_vectorizer.vocabulary_))
print("Shape:", tfidf_matrix.shape)
print("Sample TF-IDF vector (first row):", tfidf_matrix[0].toarray())

# -------------------------------
# 7. Save Outputs
# -------------------------------
# Save cleaned dataset
output_df = pd.DataFrame({
    "original_text": texts,
    "cleaned_text": cleaned_texts,
    "lemmatized_text": lemmatized_texts,
    "final_text": final_texts,
    "label": labels,
    "encoded_label": encoded_labels
})

output_df.to_csv("Assignment 3/processed_dataset.csv", index=False)

# Save TF-IDF matrix
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df.to_csv("Assignment 3/tfidf_matrix.csv", index=False)

print("\nProcessed dataset saved as 'Assignment 3/processed_dataset.csv'")
print("TF-IDF matrix saved as 'Assignment 3/tfidf_matrix.csv'")

# Close output redirection
sys.stdout.close()