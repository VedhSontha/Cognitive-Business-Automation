from app import Chatbot
import sys

def test_chatbot():
    print("Initializing Chatbot...")
    bot = Chatbot()
    
    test_queries = [
        "Hi",
        "How are you?",
        "Where is my order?", # From sample.csv (likely)
        "My internet is down", # From sample.csv
        "gibberish_text_that_should_fail"
    ]
    
    print("\nTesting queries:")
    for query in test_queries:
        response = bot.get_response(query)
        print(f"Query: '{query}'")
        print(f"Response: '{response}'")
        print("-" * 30)

    print("\nTest complete.")

if __name__ == "__main__":
    test_chatbot()
