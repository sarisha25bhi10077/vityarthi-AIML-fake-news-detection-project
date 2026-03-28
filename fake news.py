import math
import re
# 1 = Real News, 0 = Fake News
data = [
    {"headline": "Scientists discover new planet in habitable zone", "label": 1},
    {"headline": "Government to ban all internet use by 2026", "label": 0},
    {"headline": "Local hero saves cat from burning building", "label": 1},
    {"headline": "Alien invasion confirmed by secret NASA documents", "label": 0},
    {"headline": "Stock market hits record high amid economic growth", "label": 1},
    {"headline": "Drinking bleach cures all known viral diseases", "label": 0},
    {"headline": "New medical breakthrough in cancer research", "label": 1},
    {"headline": "Celebrity caught in massive offshore tax scandal", "label": 1},
    {"headline": "Moon is actually made of cheese, says new study", "label": 0},
    {"headline": "Olympic athlete breaks world record in 100m sprint", "label": 1},
    {"headline": "illegal aliens voting in large numbers in election","label":0}
]
def tokenize(text):
    return re.findall(r'\w+', text.lower())

# Build Vocabulary and calculate IDF
all_headlines = [tokenize(item["headline"]) for item in data]
vocab = sorted(list(set(word for headline in all_headlines for word in headline)))
word_to_idx = {word: i for i, word in enumerate(vocab)}

num_docs = len(all_headlines)
idf = {}
for word in vocab:
    doc_count = sum(1 for headline in all_headlines if word in headline)
    idf[word] = math.log(num_docs / (1 + doc_count))

def transform_to_tfidf(headline):
    tokens = tokenize(headline)
    vector = [0.0] * len(vocab)
    for word in tokens:
        if word in word_to_idx:
            tf = tokens.count(word) / len(tokens)
            vector[word_to_idx[word]] = tf * idf[word]
    return vector

# 3. PASSIVE-AGGRESSIVE CLASSIFIER (PAC) IMPLEMENTATION
class ManualPAC:
    def __init__(self, input_dim, C=1.0):
        self.weights = [0.0] * input_dim
        self.C = C  # Aggressiveness parameter

    def predict_raw(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x))

    def train(self, x, y):
        # y must be 1 or -1 for PAC math
        target = 1 if y == 1 else -1
        prediction = self.predict_raw(x)
        
        # Calculate Hinge Loss: max(0, 1 - y * (w . x))
        loss = max(0, 1 - target * prediction)
        
        if loss > 0:
            # Update weights: w = w + (loss / ||x||^2) * y * x
            # Restricted by C (Aggressiveness)
            sq_norm_x = sum(xi**2 for xi in x) or 1e-9
            tau = min(self.C, loss / sq_norm_x)
            
            for i in range(len(self.weights)):
                self.weights[i] += tau * target * x[i]

    def predict(self, headline):
        x = transform_to_tfidf(headline)
        return "REAL" if self.predict_raw(x) >= 0 else "FAKE"

# 4. RUNNING THE MODEL
model = ManualPAC(len(vocab), C=1.0)
for item in data:
    vec = transform_to_tfidf(item["headline"])
    model.train(vec, item["label"])
print("--- Fake News Detector (PAC from Scratch) ---")
test_headline = input("Enter a news headline to check: ")
result = model.predict(test_headline)
print(f"Prediction: {result}")