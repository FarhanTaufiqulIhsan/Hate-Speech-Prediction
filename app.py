from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load model
model = load_model("hate_speech_model.keras")

# Load tokenizer and max_sequence_length
with open("tokenizer.pkl", "rb") as handle:
    tokenizer, max_sequence_length = pickle.load(handle)


# Function to get label from prediction
def get_label(prediction):
    return "Hate Speech" if prediction > 0.5 else "Non-Hate Speech"


# Function to convert text to padded sequences
def text_to_sequence(texts, tokenizer, max_sequence_length):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_sequence_length)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from request
        data = request.json
        texts = data.get("texts", [])

        if not texts:
            return jsonify({"error": "No texts provided"}), 400

        # Convert texts to sequences
        padded_sequences = text_to_sequence(texts, tokenizer, max_sequence_length)

        # Make predictions
        predictions = model.predict(padded_sequences)
        results = [
            {
                "text": text,
                "label": get_label(pred[0]),
                "percentage": float(pred[0]) * 100,
            }
            for text, pred in zip(texts, predictions)
        ]

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
