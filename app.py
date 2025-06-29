import streamlit as st
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('next_word_lstm_tf')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

st.title("🎭 Next Word Prediction with LSTM")

st.markdown("""
Welcome to the **Next Word Predictor App!** 🚀  

📚 **How it works:**
- Enter a partial phrase or sentence from *Shakespeare's Hamlet* (or similar language) in the box below.
- The app uses a trained LSTM model to predict what the next word could be.
- Press **Predict Next Word** and see the magic! ✨

🎭 **Note:** This model was trained on *Hamlet* data. You can download the text below 📖 to find reference phrases.

💡 **Example Inputs:**
- "To be or not to"
- "What a piece of"
- "The lady doth protest too"

Try experimenting with similar phrasing for best results!
""")

with open("hamlet.txt", "r", encoding="utf-8") as file:
    hamlet_text = file.read()

st.download_button(
    label="📥 Download Hamlet Text",
    data=hamlet_text,
    file_name='hamlet.txt',
    mime='text/plain'
)

input_text = st.text_input("🖊️ Enter the sequence of words:", "To be or not to")

if st.button("🚀 Predict Next Word"):
    sequence = tokenizer.texts_to_sequences([input_text])[0]
    sequence = pad_sequences([sequence], maxlen=13, padding='pre')
    predicted_probs = model.predict(sequence, verbose=0)
    predicted_index = predicted_probs.argmax(axis=-1)[0]
    predicted_word = ''

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            predicted_word = word
            break

    if predicted_word:
        st.success(f"✅ Predicted next word: **{predicted_word}**")
    else:
        st.warning("⚠️ Could not predict the next word. Try a different sequence!")
