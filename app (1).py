import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load the model and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Set the app layout and title
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="centered")
st.title("📧 Email Spam Classifier")
st.subheader("Classify your email text as 'Spam' or 'Not Spam'")

# Style the sidebar with instructions
st.sidebar.title("🛠 Instructions")
st.sidebar.info(
    """
    - Enter the content of the email in the text box provided.
    - Click on "Classify" to see if the email is spam or not.
    """
)

# Add input box with placeholder text
st.write("Type or paste the email content below:")

user_input = st.text_area("Email Content", placeholder="Enter your email text here...", height=200)

def highlight_spam_words(text, vectorizer, model):
    # Transform the input text
    input_vectorized = vectorizer.transform([text])
    # Get the feature names (words) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()
    # Get the model coefficients (weights) for each word
    if hasattr(model, 'coef_'):
        spam_indicator_words = model.coef_[0] > 0
    else:
        spam_indicator_words = np.zeros(len(feature_names), dtype=bool)  # fallback in case model doesn't have coef_

    # Get the words from the input text that contribute to spam classification
    words_to_highlight = [
        word for word in feature_names[spam_indicator_words]
        if word in text.lower()
    ]
    
    # Highlight words by wrapping them in HTML tags
    for word in words_to_highlight:
        text = text.replace(word, f"<mark style='background-color: #ffb3b3;'>{word}</mark>")
    return text, words_to_highlight

# Add a button to classify
if st.button("Classify Email"):
    if user_input.strip() != "":
        # Process and predict
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        
        # Highlight spam words
        highlighted_text, spam_words = highlight_spam_words(user_input, vectorizer, model)
        
        # Display result with styled message
        if result == "Spam":
            st.error("🚨 The email is classified as **Spam**.")
            if spam_words:
                st.markdown(
                    f"<div style='color: red;'>Spam words detected in email: {' '.join(spam_words)}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(f"<div>{highlighted_text}</div>", unsafe_allow_html=True)
        else:
            st.success("✅ The email is classified as **Not Spam**.")
    else:
        st.warning("⚠️ Please enter some text in the email content area.")

# Footer
st.markdown(
    """
    <style>
        .footer {font-size:12px; text-align: center; color: grey; padding-top: 20px;}
    </style>
    <div class="footer">Developed with ❤️ by Omprakash </div>
    """,
    unsafe_allow_html=True,
)
