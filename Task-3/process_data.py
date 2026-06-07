import pandas as pd
import json
import os

def process_data():
    qa_pairs = []

    # 1. Process dialogs.txt (Chitchat)
    dialogs_path = 'dialogs.txt'
    if os.path.exists(dialogs_path):
        print(f"Processing {dialogs_path}...")
        with open(dialogs_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    question = parts[0]
                    answer = parts[1]
                    qa_pairs.append({'question': question, 'answer': answer, 'source': 'dialogs'})
    else:
        print(f"Warning: {dialogs_path} not found.")

    # 2. Process sample.csv (Customer Support)
    csv_path = 'sample.csv'
    if os.path.exists(csv_path):
        print(f"Processing {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Ensure tweet_id and response_tweet_id are strings to avoid type mismatches
        df['tweet_id'] = df['tweet_id'].astype(str)
        df['response_tweet_id'] = df['response_tweet_id'].astype(str)
        
        # Filter for inbound tweets (customer questions) that have a response
        customer_tweets = df[df['inbound'] == True].dropna(subset=['response_tweet_id'])
        
        for index, row in customer_tweets.iterrows():
            question = row['text']
            # response_tweet_id can be comma separated, take the first one
            response_ids = row['response_tweet_id'].split(',')
            
            for response_id in response_ids:
                response_row = df[df['tweet_id'] == response_id.strip()]
                if not response_row.empty:
                    answer = response_row.iloc[0]['text']
                    qa_pairs.append({'question': question, 'answer': answer, 'source': 'customer_support'})
    else:
        print(f"Warning: {csv_path} not found.")

    # Save to JSON
    output_path = 'knowledge_base.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=4)
    
    print(f"Successfully saved {len(qa_pairs)} QA pairs to {output_path}")

if __name__ == "__main__":
    process_data()
