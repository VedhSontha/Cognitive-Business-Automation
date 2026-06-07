import streamlit as st
import json
import random
from sentence_transformers import SentenceTransformer, util
import torch

class Chatbot:
    def __init__(self, knowledge_base_path='knowledge_base.json'):
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        self.questions = [item['question'] for item in self.knowledge_base]
        
        # Load Sentence Transformer model
        # Using a small, fast model
        with st.spinner("Loading AI Brain... (this might take a minute initially)"):
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
        # Pre-compute embeddings for all questions
        if self.questions:
            self.question_embeddings = self.model.encode(self.questions, convert_to_tensor=True)
        else:
            self.question_embeddings = None

    def load_knowledge_base(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            st.error(f"Knowledge base file not found at {path}. Please run process_data.py first.")
            return []

    def get_response(self, user_input, threshold=0.4):
        if self.question_embeddings is None:
            return "I'm not initialized properly.", None, 0.0

        # Encode user input
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)

        # Calculate cosine similarity
        cosine_scores = util.cos_sim(user_embedding, self.question_embeddings)[0]
        
        # Find best match
        best_match_index = torch.argmax(cosine_scores).item()
        best_match_score = cosine_scores[best_match_index].item()

        if best_match_score > threshold:
            return self.knowledge_base[best_match_index]['answer'], self.knowledge_base[best_match_index]['question'], best_match_score
        else:
            return self.get_fallback_response(), None, best_match_score

    def get_fallback_response(self):
        fallbacks = [
            "I'm sorry, I didn't quite understand that. Could you rephrase?",
            "I don't have an answer for that right now. You might want to contact support directly.",
            "That's a bit beyond my knowledge. Can you try asking something else?",
            "I'm still learning! Try asking about store hours, order status, or general questions."
        ]
        return random.choice(fallbacks)

import csv
import os
from datetime import datetime

# ... (Chatbot class remains the same)

def log_ticket(name, issue):
    file_exists = os.path.isfile('support_tickets.csv')
    with open('support_tickets.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Name', 'Issue'])
        writer.writerow([datetime.now(), name, issue])

# Streamlit UI
def main():
    st.set_page_config(page_title="Smart Customer Support Chatbot", page_icon="🧠")

    st.title("🧠 Smart Customer Support Chatbot")
    st.markdown("Welcome! I use **Semantic Search** to understand what you mean, not just what you say.")

    # Initialize Chatbot (cached)
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = Chatbot()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi there! How can I help you today?"}]

    # Sidebar options
    st.sidebar.header("Settings")
    # Higher default threshold for semantic search as it's more confident
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False, help="Show what the bot matched with.")

    st.sidebar.markdown("---")
    st.sidebar.header("Suggested Questions")
    
    if st.session_state.chatbot.knowledge_base:
        # Get a few random questions
        sample_questions = random.sample([q['question'] for q in st.session_state.chatbot.knowledge_base], min(5, len(st.session_state.chatbot.knowledge_base)))
        for q in sample_questions:
            st.sidebar.code(q)

    st.sidebar.markdown("---")
    with st.sidebar.expander("📩 Contact Support"):
        st.write("Can't find an answer? Submit a ticket.")
        with st.form("ticket_form"):
            name = st.text_input("Your Name")
            issue = st.text_area("Describe your issue")
            submitted = st.form_submit_button("Submit Ticket")
            if submitted:
                if name and issue:
                    log_ticket(name, issue)
                    st.success("Ticket submitted! We'll contact you soon.")
                else:
                    st.error("Please fill in all fields.")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get bot response
        response, matched_question, score = st.session_state.chatbot.get_response(prompt, threshold=confidence_threshold)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
            if debug_mode and matched_question:
                st.info(f"**Matched:** '{matched_question}'\n\n**Score:** {score:.4f}")
            elif debug_mode:
                st.warning(f"**No match found.** Best score: {score:.4f}")
            
            # If confidence is low, suggest submitting a ticket
            if score < confidence_threshold:
                st.markdown("---")
                st.markdown("**Did I miss the mark?** You can submit a support ticket in the sidebar 📩")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
