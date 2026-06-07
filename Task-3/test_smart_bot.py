from app import Chatbot
import time

def test_smart_bot():
    print("Initializing Smart Chatbot (this may take a moment)...")
    start = time.time()
    bot = Chatbot()
    print(f"Initialization took {time.time() - start:.2f} seconds.")
    
    test_queries = [
        "are you ok",        # The problematic query
        "how are you doing", # Basic greeting
        "internet down",     # Tech support (fuzzy)
        "my phone battery dies fast" # Tech support (semantic)
    ]
    
    print("\nTesting queries:")
    for query in test_queries:
        response, matched_q, score = bot.get_response(query)
        print(f"Query: '{query}'")
        print(f"Matched: '{matched_q}' (Score: {score:.4f})")
        print(f"Response: '{response}'")
        print("-" * 30)

if __name__ == "__main__":
    test_smart_bot()
