from flask import Flask, render_template, request
import joblib


app = Flask(__name__)


# ============================================================
# LOAD MODELS
# ============================================================

language_model = joblib.load(
    "models/language_model.pkl"
)

language_vectorizer = joblib.load(
    "models/language_vectorizer.pkl"
)

category_model = joblib.load(
    "models/category_model.pkl"
)

category_vectorizer = joblib.load(
    "models/category_vectorizer.pkl"
)


# ============================================================
# RESPONSE MESSAGES
# ============================================================

responses = {

    "Health":
        "You seem to be expressing a health-related concern.",

    "Mood":
        "You seem to be expressing a concern about your mood.",

    "Work/Office":
        "You are informing that you have a work or office-related matter.",

    "Travel":
        "You seem to be talking about a travel-related matter.",

    "Complaint":
        "You seem to be reporting a complaint or service-related issue.",

    "General":
        "You seem to be asking for general information or clarification."
}


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    if request.method == "POST":

        text = request.form.get("text", "").strip()

        # Input validation
        if not text:

            error = "Please enter some text."

            return render_template(
                "index.html",
                error=error
            )

        try:

            # ------------------------------------------------
            # LANGUAGE PREDICTION
            # ------------------------------------------------

            language_features = language_vectorizer.transform(
                [text]
            )

            language_prediction = language_model.predict(
                language_features
            )[0]

            language_confidence = max(
                language_model.predict_proba(
                    language_features
                )[0]
            )


            # ------------------------------------------------
            # CATEGORY PREDICTION
            # ------------------------------------------------

            category_features = category_vectorizer.transform(
                [text]
            )

            category_prediction = category_model.predict(
                category_features
            )[0]

            category_confidence = max(
                category_model.predict_proba(
                    category_features
                )[0]
            )


            # ------------------------------------------------
            # RESPONSE
            # ------------------------------------------------

            generated_response = responses.get(
                category_prediction,
                "Your message has been classified."
            )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            result = {

                "text": text,

                "language": language_prediction,

                "language_confidence":
                    round(
                        float(language_confidence),
                        4
                    ),

                "category": category_prediction,

                "confidence":
                    round(
                        float(category_confidence),
                        4
                    ),

                "response": generated_response
            }


        except Exception as e:

            error = f"Prediction error: {str(e)}"


    return render_template(
        "index.html",
        result=result,
        error=error
    )


# ============================================================
# API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    from flask import jsonify

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


    # Language

    language_features = language_vectorizer.transform(
        [text]
    )

    language = language_model.predict(
        language_features
    )[0]


    # Category

    category_features = category_vectorizer.transform(
        [text]
    )

    category = category_model.predict(
        category_features
    )[0]


    confidence = max(
        category_model.predict_proba(
            category_features
        )[0]
    )


    response = responses.get(
        category,
        "Your message has been classified."
    )


    return jsonify({

        "category": category,

        "confidence": round(
            float(confidence),
            4
        ),

        "response": response,

        "language": language
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )