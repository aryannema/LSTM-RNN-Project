# Next Word Prediction with LSTM 🎭

This project is a Streamlit web app that predicts the next word in a sequence using an LSTM model trained on Shakespeare's _Hamlet_ text.

## 🚀 Features

- Predict the next word for partial sentences.
- Provides example phrases and tips to help users get started.
- Includes a downloadable _Hamlet_ text file for reference.
- User-friendly interface with emojis and helpful guidance.

## 📚 How to Use

1. Run the app with:
   ```bash
   streamlit run app.py
   ```
2. Enter a partial sentence in the input box. Example: "To be or not to".
3. Click the **Predict Next Word** button to see the prediction.
4. Download the _Hamlet_ text for ideas using the **Download Hamlet Text** button.

## 🗂️ Files

- `app.py`: Main Streamlit app.
- `next_word_lstm_tf`: Saved LSTM model directory.
- `tokenizer.pkl`: Pickled tokenizer object.
- `hamlet.txt`: Reference text data.

## 🔧 Requirements

- Python 3.8+
- `tensorflow`
- `keras`
- `streamlit`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 🎯 Note

This model was trained specifically on _Hamlet_, so best results come from using phrases similar to Shakespearean language.

---
