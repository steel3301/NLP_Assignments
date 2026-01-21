# Assignment 1: Tokenization, Stemming, Lemmatization on Custom Dataset
import sys
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, WhitespaceTokenizer, WordPunctTokenizer, TreebankWordTokenizer, TweetTokenizer, MWETokenizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

sys.stdout = open("output.txt", "w", encoding="utf-8")

# Download required NLTK resources
nltk.download('punkt', quiet = True)
nltk.download('wordnet', quiet = True)
nltk.download('omw-1.4', quiet = True)

data = {
    "id": [1, 2, 3],
    "text": [
        "I am playing Borderlands and loving it!",
        "Borderlands 2 is amazing, but sometimes frustrating.",
        "I will murder you all in Borderlands (just kidding, lol)."
    ]
}

df = pd.DataFrame(data)
print("\n--- Custom Dataset ---")
print(df)

# -------------------------------
# 2. Tokenization
# -------------------------------
print("\n--- TOKENIZATION ---")

sample_text = df.loc[0, "text"]

# Whitespace Tokenizer
whitespace_tokens = WhitespaceTokenizer().tokenize(sample_text)
print("Whitespace Tokens:", whitespace_tokens)

# Punctuation-based Tokenizer
punct_tokens = WordPunctTokenizer().tokenize(sample_text)
print("Punctuation-based Tokens:", punct_tokens)

# Treebank Tokenizer
treebank_tokens = TreebankWordTokenizer().tokenize(sample_text)
print("Treebank Tokens:", treebank_tokens)

# Tweet Tokenizer
tweet_tokens = TweetTokenizer().tokenize(sample_text)
print("Tweet Tokens:", tweet_tokens)

# MWE Tokenizer (Multi-Word Expressions)
mwe_tokenizer = MWETokenizer([('Borderlands', '2')])
mwe_tokens = mwe_tokenizer.tokenize(word_tokenize(sample_text))
print("MWE Tokens:", mwe_tokens)

# -------------------------------
# 3. Stemming
# -------------------------------
print("\n--- STEMMING ---")

porter = PorterStemmer()
snowball = SnowballStemmer("english")

porter_stems = [porter.stem(word) for word in whitespace_tokens]
snowball_stems = [snowball.stem(word) for word in whitespace_tokens]

print("Porter Stemmer:", porter_stems)
print("Snowball Stemmer:", snowball_stems)

# -------------------------------
# 4. Lemmatization
# -------------------------------
print("\n--- LEMMATIZATION ---")

lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word) for word in whitespace_tokens]
print("WordNet Lemmatizer:", lemmas)

# -------------------------------
# 5. Apply to Entire Dataset
# -------------------------------
print("\n--- APPLYING TO ENTIRE DATASET ---")
results = []

for i, row in df.iterrows():
    tokens = word_tokenize(row["text"])
    porter_stems = [porter.stem(w) for w in tokens]
    snowball_stems = [snowball.stem(w) for w in tokens]
    lemmas = [lemmatizer.lemmatize(w) for w in tokens]
    
    results.append({
        "id": row["id"],
        "text": row["text"],
        "tokens": tokens,
        "porter_stems": porter_stems,
        "snowball_stems": snowball_stems,
        "lemmas": lemmas
    })

results_df = pd.DataFrame(results)
print(results_df)

sys.stdout.close()