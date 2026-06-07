from sklearn.feature_extraction.text import TfidfVectorizer

def test_vectorizer():
    corpus = [
        "how are you doing?",
        "what is your name?",
        "i am fine"
    ]
    
    # Current implementation
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform(corpus)
        print(f"Vocabulary with stop_words='english': {vectorizer.get_feature_names_out()}")
    except ValueError as e:
        print(f"Error with stop_words='english': {e}")

    query = "how are you"
    try:
        vec = vectorizer.transform([query])
        print(f"Vector for '{query}': {vec.sum()}")
    except Exception as e:
        print(f"Error transforming '{query}': {e}")

    # Proposed fix
    print("\n--- Proposed Fix ---")
    vectorizer_fix = TfidfVectorizer(stop_words=None) # or just default
    X_fix = vectorizer_fix.fit_transform(corpus)
    print(f"Vocabulary without stop_words: {vectorizer_fix.get_feature_names_out()}")
    vec_fix = vectorizer_fix.transform([query])
    print(f"Vector for '{query}' (fixed): {vec_fix.sum()}")

if __name__ == "__main__":
    test_vectorizer()
