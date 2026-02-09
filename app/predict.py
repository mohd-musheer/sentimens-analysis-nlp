import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Load pipeline
pipeline = joblib.load("model/mental_health_model.pkl")

# Example new text (raw, not preprocessed)
new_text = "i am feeling too lonely"
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    # lowercase
    text = text.lower()                          # lowercase
    text = re.sub(r'http\S+|www\S+', '', text)   # remove urls
    text = re.sub(r'\S+@\S+', '', text)          # remove emails
    text = re.sub(r'[^a-z\s]', '', text)         # remove numbers & special chars
    text = re.sub(r'\s+', ' ', text).strip()     # remove extra spaces

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords & lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# Preprocess (same as training)
clean_text = preprocess_text(new_text)

# Predict 
pred_encoded = pipeline.predict([clean_text])[0]

# Decode back to category
reverse_mapping = {0:'Anxiety', 1:'Bipolar', 2:'Depression', 3:'Normal',
                   4:'Personality_disorder', 5:'Stress', 6:'Suicidal'}

print("Predicted category:", reverse_mapping[pred_encoded])

