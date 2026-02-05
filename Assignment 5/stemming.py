import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import (
    WhitespaceTokenizer,
    wordpunct_tokenize,
    TreebankWordTokenizer,
    TweetTokenizer,
    MWETokenizer
)
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

# Sample text
text = "NLTK is great! I love Natural Language Processing :) #NLP @OpenAI"
mwe_text = "I live in New York City and work on machine learning."

print("Original Text:")
print(text)

# ---------------- TOKENIZATION ----------------

# 1. Whitespace Tokenization
wt = WhitespaceTokenizer()
print("\nWhitespace Tokenization:")
print(wt.tokenize(text))

# 2. Punctuation-based Tokenization
print("\nPunctuation-based Tokenization:")
print(wordpunct_tokenize(text))

# 3. Treebank Tokenization
tbt = TreebankWordTokenizer()
print("\nTreebank Tokenization:")
print(tbt.tokenize(text))

# 4. Tweet Tokenization
tweet_tokenizer = TweetTokenizer()
print("\nTweet Tokenization:")
print(tweet_tokenizer.tokenize(text))

# 5. MWE Tokenization
mwe_tokenizer = MWETokenizer([('new', 'york', 'city'), ('machine', 'learning')], separator='_')
print("\nMWE Tokenization:")
print(mwe_tokenizer.tokenize(mwe_text.lower().split()))

# ---------------- STEMMING ----------------

words = ["running", "runs", "runner", "easily", "fairly"]

# Porter Stemmer
porter = PorterStemmer()
print("\nPorter Stemmer:")
for word in words:
    print(word, "->", porter.stem(word))

# Snowball Stemmer
snowball = SnowballStemmer("english")
print("\nSnowball Stemmer:")
for word in words:
    print(word, "->", snowball.stem(word))

# ---------------- LEMMATIZATION ----------------

lemmatizer = WordNetLemmatizer()
lemmatization_words = ["running", "better", "cars", "went"]

print("\nLemmatization:")
for word in lemmatization_words:
    print(word, "->", lemmatizer.lemmatize(word))
