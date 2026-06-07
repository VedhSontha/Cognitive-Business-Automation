import json
from collections import Counter

def analyze_kb():
    with open('knowledge_base.json', 'r', encoding='utf-8') as f:
        kb = json.load(f)
    
    total = len(kb)
    sources = Counter(item['source'] for item in kb)
    
    print(f"Total Questions: {total}")
    print(f"Sources: {dict(sources)}")
    
    print("\n--- Customer Support Examples (Random 10) ---")
    support_qs = [item['question'] for item in kb if item['source'] == 'customer_support']
    import random
    if support_qs:
        for q in random.sample(support_qs, min(10, len(support_qs))):
            print(f"- {q}")
            
    print("\n--- Chitchat Examples (Random 5) ---")
    chitchat_qs = [item['question'] for item in kb if item['source'] == 'dialogs']
    if chitchat_qs:
        for q in random.sample(chitchat_qs, min(5, len(chitchat_qs))):
            print(f"- {q}")

if __name__ == "__main__":
    analyze_kb()
