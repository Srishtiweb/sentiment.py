import nltk
nltk.download('vader_lexico')
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Example sentences for sentiment analysis
sentences = [
    "I love this product!",
    "The movie was terrible.",
    "The weather is nice today.",
    "I feel neutral about this situation.",
    "This restaurant has delicious food.",
    "The customer service was disappointing."
]

# Perform sentiment analysis on each sentence
for sentence in sentences:
    sentiment_score = sia.polarity_scores(sentence)
    print("Sentence:", sentence)
    print("Sentiment Score:", sentiment_score)
    print("Sentiment Label:", sentiment_score['compound'])
    print()
