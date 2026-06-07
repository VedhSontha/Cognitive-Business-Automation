from app import log_ticket
import os
import csv

def test_logging():
    print("Testing ticket logging...")
    log_ticket("Test User", "This is a test issue")
    
    if os.path.exists('support_tickets.csv'):
        print("Success: support_tickets.csv created.")
        with open('support_tickets.csv', 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            print(f"Rows found: {len(rows)}")
            print(f"Last row: {rows[-1]}")
    else:
        print("Error: File not created.")

if __name__ == "__main__":
    test_logging()
