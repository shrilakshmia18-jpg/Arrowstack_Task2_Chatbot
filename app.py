from flask import Flask, render_template, request, jsonify

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


# =====================================================
# CHATBOT KNOWLEDGE BASE
# =====================================================

questions = [
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "what is your name",
    "who are you",
    "what can you do",
    "what is artificial intelligence",
    "what is machine learning",
    "what is computer vision",
    "how can I learn python",
    "how can I learn AI",
    "what courses are available",
    "thank you",
    "thanks",
    "bye",
    "goodbye"
]


answers = [
    "Hello! 👋 How can I help you?",
    "Hi! 👋 How can I help you today?",
    "Hey! 😊 What would you like to know?",
    "Good morning! ☀️ How can I help you?",
    "Good afternoon! 😊 How can I help you?",
    "I am an AI-powered support chatbot.",
    "I am a rule-based and semantic chatbot developed using Python and machine learning techniques.",
    "I can answer questions about AI, Machine Learning, Computer Vision, Python, and learning resources.",
    "Artificial Intelligence is the field of creating systems that can perform tasks that normally require human intelligence.",
    "Machine Learning is a branch of AI where computers learn patterns from data to make predictions or decisions.",
    "Computer Vision is a field of AI that enables computers to understand and analyze images and videos.",
    "You can start Python by learning variables, data types, conditions, loops, functions, and basic projects.",
    "You can learn AI by starting with Python, Data Analytics, SQL, Machine Learning, Deep Learning, and Generative AI.",
    "Recommended learning areas include Python, Data Analytics, SQL, Machine Learning, Deep Learning, and Generative AI.",
    "You're welcome! 😊",
    "You're welcome! 😊",
    "Goodbye! 👋 Have a great day!",
    "Goodbye! 👋 See you again!"
]


# =====================================================
# SEMANTIC MODEL
# =====================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

question_vectors = vectorizer.fit_transform(questions)


# =====================================================
# CHATBOT LOGIC
# =====================================================

def get_response(user_input):

    text = user_input.lower().strip()

    # Rule-based responses

    if text in ["exit", "quit", "stop"]:
        return "Goodbye! 👋"

    if "help" in text:
        return (
            "I can help you with:\n"
            "• Artificial Intelligence\n"
            "• Machine Learning\n"
            "• Computer Vision\n"
            "• Python\n"
            "• AI learning roadmap"
        )

    # Semantic matching

    user_vector = vectorizer.transform([text])

    similarity_scores = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[0][best_match_index]

    if best_score >= 0.25:

        return answers[best_match_index]

    return (
        "Sorry, I don't have a good answer for that yet. "
        "Try asking about AI, Machine Learning, "
        "Computer Vision, Python, or learning resources."
    )


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():

    return render_template("index.html")


# =====================================================
# CHAT API
# =====================================================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_message = data.get("message", "").strip()

    if not user_message:

        return jsonify({
            "response": "Please enter a message."
        })

    bot_response = get_response(user_message)

    return jsonify({
        "response": bot_response
    })


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )