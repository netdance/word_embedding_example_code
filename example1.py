from gensim.models import Word2Vec

# Example corpus
sentences = [["cat", "sat", "on", "the", "mat"],
             ["dog", "barked", "at", "the", "mailman"],
             ["dog", "chased", "the", "cat"]]

print("training models")
# Train a Word2Vec model
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, sg=1)
print("model trained")
# Get the vector for 'cat'
vector_cat = model.wv['cat']
print(vector_cat)

print(model.corpus_total_words)

print(model.predict_output_word(['cat']))