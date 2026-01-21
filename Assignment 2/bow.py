import sys
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import nltk

# Redirect all prints to output.txt in Assignment 2 folder
sys.stdout = open("Assignment 2/output.txt", "w", encoding="utf-8")


data_path = "data/train.jsonl"   # <-- modified dataset path

records = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

# Use the 'messages' field as text
texts = df['messages'].dropna().astype(str).tolist()

print("\n--- Dataset Loaded ---")
print("Number of documents:", len(texts))
print("Sample text:", texts[0])


count_vectorizer = CountVectorizer()
bow_counts = count_vectorizer.fit_transform(texts)

print("\n--- Bag-of-Words (Count Occurrence) ---")
print("Vocabulary size:", len(count_vectorizer.vocabulary_))
print("Shape:", bow_counts.shape)
print("Sample vector (first row):", bow_counts[0].toarray())


from sklearn.preprocessing import normalize

bow_counts_norm = normalize(bow_counts, norm='l2')

print("\n--- Bag-of-Words (Normalized Count Occurrence) ---")
print("Shape:", bow_counts_norm.shape)
print("Sample normalized vector (first row):", bow_counts_norm[0].toarray())


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

print("\n--- TF-IDF ---")
print("Vocabulary size:", len(tfidf_vectorizer.vocabulary_))
print("Shape:", tfidf_matrix.shape)
print("Sample TF-IDF vector (first row):", tfidf_matrix[0].toarray())


nltk.download('punkt', quiet=True)

tokenized_texts = [nltk.word_tokenize(doc.lower()) for doc in texts]

w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4)

print("\n--- Word2Vec Embeddings ---")
print("Vocabulary size:", len(w2v_model.wv))

# Example: vector for 'russia' (since dataset is Diplomacy)
if 'russia' in w2v_model.wv:
    print("Vector for 'russia':\n", w2v_model.wv['russia'])
    print("\nMost similar to 'russia':")
    print(w2v_model.wv.most_similar('russia'))
else:
    print("'russia' not found in vocabulary.")

# Close output redirection
sys.stdout.close()