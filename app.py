import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Customer Support Chatbot", layout="centered")

st.title("🤖 AI Customer Support Chatbot")

# Load the Hugging Face pipelines
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad"
)
sentiment_pipeline = pipeline("sentiment-analysis")

st.write("Ask a question and I’ll do my best to help!")

user_input = st.text_input("Your message:")

if user_input:
    # Sentiment
    sentiment = sentiment_pipeline(user_input)[0]
    st.write(f"**Sentiment:** {sentiment['label']} ({round(sentiment['score'] * 100)}%)")

    # Use a dummy context (real system would use FAQs or documents)
    context = """
    Our company offers 24/7 customer support. You can reset your password from the login page.
    Shipping takes 3–5 days in Europe. Refunds are available within 30 days of purchase.
    """
    response = qa_pipeline(question=user_input, context=context)

    st.write("**Answer:**", response["answer"])
