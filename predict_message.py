import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import urllib.request
import io
import os

# Beberapa alternatif URL untuk dataset SMS Spam
urls = [
    'https://raw.githubusercontent.com/freeCodeCamp/boilerplate-neural-network-sms-text-classifier/master/spam.csv',
    'https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset/master/spam.csv',
    'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv'
]

# Coba download dari berbagai sumber
def download_dataset():
    for url in urls:
        try:
            print(f"Mencoba download dari: {url}")
            response = urllib.request.urlopen(url)
            data = response.read().decode('utf-8')
            return data
        except Exception as e:
            print(f"Gagal: {e}")
            continue
    
    # Jika semua URL gagal, gunakan dataset lokal atau buat manual
    print("Menggunakan dataset fallback...")
    return create_fallback_dataset()

def create_fallback_dataset():
    """Membuat dataset sederhana jika download gagal"""
    data = """label,text
ham,Hello how are you today
ham,Can we meet tomorrow
ham,Thanks for your help
ham,What time is the meeting
spam,Free gift card click now
spam,You won $1000 call immediately
spam,Urgent your account will be suspended
spam,Congratulations you are a winner"""
    return data

# Download dataset
print("Mengunduh dataset...")
data = download_dataset()

# Load dataset
try:
    dataset = pd.read_csv(io.StringIO(data), encoding='ISO-8859-1')
except:
    # Jika format CSV gagal, coba format TSV
    try:
        dataset = pd.read_csv(io.StringIO(data), sep='\t', names=['label', 'text'], encoding='ISO-8859-1')
    except:
        # Gunakan dataset fallback
        dataset = pd.read_csv(io.StringIO(create_fallback_dataset()))

print(f"Dataset loaded: {dataset.shape}")
print(dataset.head())

# Preprocess data
# Bersihkan kolom yang tidak diperlukan dan pastikan formatnya benar
dataset = dataset.dropna()
if len(dataset.columns) > 2:
    # Ambil hanya 2 kolom pertama jika ada lebih banyak kolom
    dataset = dataset.iloc[:, :2]
    dataset.columns = ['label', 'text']

dataset.columns = ['label', 'text']

# Pastikan label dalam format yang benar
if dataset['label'].dtype == 'object':
    dataset['label'] = dataset['label'].map({'ham': 0, 'spam': 1, '0': 0, '1': 1})
else:
    # Jika sudah numerik, pastikan 0=ham, 1=spam
    dataset['label'] = dataset['label'].apply(lambda x: 0 if x == 0 else 1)

print(f"Setelah preprocessing: {dataset.shape}")
print(f"Distribusi label: {dataset['label'].value_counts()}")

# Split data
X = dataset['text'].astype(str)
y = dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Create tokenizer
def create_tokenizer(texts):
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer

tokenizer = create_tokenizer(X_train)
print(f"Vocabulary size: {len(tokenizer.word_index)}")

# Encode texts
def encode_texts(texts, tokenizer, max_len=100):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded

X_train_encoded = encode_texts(X_train, tokenizer)
X_test_encoded = encode_texts(X_test, tokenizer)

print(f"Encoded training shape: {X_train_encoded.shape}")

# Build model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=5000, output_dim=16, input_length=100),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train model
print("Training model...")
history = model.fit(X_train_encoded,
                    y_train,
                    epochs=15,
                    validation_data=(X_test_encoded, y_test),
                    verbose=1,
                    batch_size=32)

# Prediction function
def predict_message(message):
    try:
        # Preprocess input
        encoded_message = encode_texts([message], tokenizer)
        
        # Make prediction
        prediction = model.predict(encoded_message, verbose=0)
        
        # Get results
        probability = float(prediction[0][0])
        label = "spam" if probability > 0.5 else "ham"
        
        # Return probability for spam and label
        return [probability, label]
    except Exception as e:
        print(f"Error in prediction: {e}")
        return [0.5, "ham"]  # Return neutral prediction if error

# Test the function
if __name__ == "__main__":
    test_messages = [
        "Hello, how are you today?",
        "Congratulations! You've won a $1000 gift card! Call now!",
        "Can we meet tomorrow for coffee?",
        "URGENT: Your account will be suspended. Click here to verify.",
        "Thanks for your help with the project.",
        "Free money now!!! Click here",
        "Hey, what's up for dinner tonight?"
    ]
    
    print("\n" + "="*50)
    print("TESTING PREDICTIONS")
    print("="*50)
    
    for msg in test_messages:
        result = predict_message(msg)
        print(f"Message: {msg}")
        print(f"Prediction: {result[1]} (confidence: {result[0]:.4f})")
        print("-" * 30)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test_encoded, y_test, verbose=0)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print(f"Model Loss: {loss:.4f}")