import streamlit as st
import requests


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Smart Text Classifier",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("🤖 Smart Text Response Classifier")

st.write(
    "Enter an English or Hinglish sentence and "
    "the ML model will classify it."
)


# ==========================================
# TEXT INPUT
# ==========================================

text = st.text_area(
    "Enter your text",
    placeholder="Example: Mujhe office late pahuchna hai",
    height=150
)


# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button("Predict", type="primary"):

    if not text.strip():

        st.error("Please enter some text.")

    else:

        try:

            with st.spinner("Analyzing text..."):

                response = requests.post(
                    "http://127.0.0.1:5000/predict",
                    json={
                        "text": text
                    },
                    timeout=10
                )


            # ------------------------------
            # Successful response
            # ------------------------------

            if response.status_code == 200:

                result = response.json()


                st.success("Prediction completed!")


                # --------------------------
                # Results
                # --------------------------

                col1, col2 = st.columns(2)


                with col1:

                    st.metric(
                        "Language",
                        result["language"]
                    )

                    st.metric(
                        "Category",
                        result["category"]
                    )


                with col2:

                    st.metric(
                        "Language Confidence",
                        f"{result['language_confidence'] * 100:.2f}%"
                    )

                    st.metric(
                        "Category Confidence",
                        f"{result['category_confidence'] * 100:.2f}%"
                    )


                # --------------------------
                # Response
                # --------------------------

                st.subheader("Generated Response")

                st.info(
                    result["response"]
                )


            else:

                error = response.json()

                st.error(
                    error.get(
                        "error",
                        "Something went wrong."
                    )
                )


        except requests.exceptions.ConnectionError:

            st.error(
                "Unable to connect to Flask API. "
                "Please make sure Flask is running."
            )


        except requests.exceptions.Timeout:

            st.error(
                "Request timed out. Please try again."
            )


        except Exception as e:

            st.error(
                f"Unexpected error: {str(e)}"
            )