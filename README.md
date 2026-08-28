# Smart Text Response Classifier

A Flask-based ML application that predicts:

* **Language:** English / Hinglish
* **Category:** Health, Mood, Work/Office, Travel, Complaint, General
* **Language Confidence Score**
* **Category Confidence Score**
* Predefined response

## Tech Stack

* Python **3.11**
* Flask
* Scikit-learn
* TF-IDF
* Naive Bayes
* Logistic Regression
* Docker

## Project Setup

### 1. Clone Repository

```bash
git clone https://github.com/Shibansiddiqui/ml-flask-assignment.git
cd ml-flask-assignment
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Docker

Build:

```bash
docker build -t shibansiddiqui/sentence_classifier:latest .
```

Run:

```bash
docker run -p 5000:5000 shibansiddiqui/sentence_classifier:latest
```

Open:

```text
http://localhost:5000
```
