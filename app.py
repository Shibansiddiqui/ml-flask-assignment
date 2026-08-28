from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

language_model = joblib.load("models/language_model.pkl")
category_model = joblib.load("models/category_model.pkl")

responses = {
    "Health": "You seem to be expressing a health-related concern.",
    "Mood": "You seem to be expressing a concern about your mood.",
    "Work/Office": "You are informing that you have a work or office-related matter.",
    "Travel": "You seem to be talking about a travel-related matter.",
    "Complaint": "You seem to be reporting a complaint or service-related issue.",
    "General": "You seem to be asking for general information or clarification."
}
def predict_text(text):

    language = language_model.predict([text])[0]
    language_confidence = float(max(
        language_model.predict_proba([text])[0]
    ))

    category = category_model.predict([text])[0]
    category_confidence = float(max(
        category_model.predict_proba([text])[0]
    ))

    response = responses.get(
        category,
        "Your message has been classified."
    )

    return {
        "language": language,
        "language_confidence": round(float(language_confidence), 4),
        "category": category,
        "confidence": round(float(category_confidence), 4),
        "response": response
    }

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    if request.method == "POST":

        text = request.form.get("text", "").strip()

        if not text:
            error = "Please enter some text."

        else:
            try:
                result = predict_text(text)
                result["text"] = text

            except Exception as e:
                error = f"Prediction error: {str(e)}"

    return render_template(
        "index.html",
        result=result,
        error=error
    )


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Request body is required."
        }), 400

    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "error": "Please enter some text."
        }), 400

    try:
        result = predict_text(text)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": f"Prediction error: {str(e)}"
        }), 500

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )