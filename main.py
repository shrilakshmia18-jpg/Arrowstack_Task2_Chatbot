from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# CHATBOT KNOWLEDGE BASE
# -------------------------------------------------

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
    "I am a rule-based and semantic chatbot designed to answer common questions.",
    "I can answer questions about AI, machine learning, computer vision, Python, and learning resources.",
    "Artificial Intelligence is the field of creating systems that can perform tasks that normally require human intelligence.",
    "Machine Learning is a branch of AI where computers learn patterns from data to make predictions or decisions.",
    "Computer Vision is a field of AI that enables computers to understand and analyze images and videos.",
    "You can start Python by learning variables, data types, conditions, loops, functions, and basic projects.",
    "You can learn AI by starting with Python, data analysis, machine learning, deep learning, and practical projects.",
    "The recommended learning areas are Python, Data Analytics, SQL, Machine Learning, Deep Learning, and Generative AI.",
    "You're welcome! 😊",
    "You're welcome! 😊",
    "Goodbye! 👋 Have a great day!",
    "Goodbye! 👋 See you again!"
]


# -------------------------------------------------
# TF-IDF SEMANTIC MODEL
# -------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

question_vectors = vectorizer.fit_transform(questions)


# -------------------------------------------------
# RULE-BASED RESPONSES
# -------------------------------------------------

def rule_based_response(user_input):

    text = user_input.lower().strip()

    if text in ["exit", "quit", "stop"]:
        return "EXIT"

    if "help" in text:
        return (
            "I can help with:\n"
            "• Artificial Intelligence\n"
            "• Machine Learning\n"
            "• Computer Vision\n"
            "• Python\n"
            "• AI learning roadmap"
        )

    return None


# -------------------------------------------------
# SEMANTIC RESPONSE
# -------------------------------------------------

def semantic_response(user_input):

    user_vector = vectorizer.transform([user_input])

    similarity_scores = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[0][best_match_index]

    # Confidence threshold
    if best_score >= 0.25:
        return answers[best_match_index]

    return (
        "Sorry, I don't have a good answer for that yet. "
        "Please try asking about AI, Machine Learning, "
        "Computer Vision, Python, or learning resources."
    )


# -------------------------------------------------
# CHAT FUNCTION
# -------------------------------------------------

def chatbot():

    print("\n" + "=" * 55)
    print("        ARROWSTACK AI CHATBOT")
    print("=" * 55)

    print("Type your question below.")
    print("Type 'help' for available topics.")
    print("Type 'exit' to close the chatbot.")
    print("=" * 55)

    while True:

        user_input = input("\nYou: ")

        # Rule-based response
        rule_response = rule_based_response(user_input)

        if rule_response == "EXIT":
            print("Bot: Goodbye! 👋")
            break

        if rule_response:
            print("Bot:", rule_response)
            continue

        # Semantic response
        response = semantic_response(user_input)

        print("Bot:", response)


# -------------------------------------------------
# START CHATBOT
# -------------------------------------------------

if __name__ == "__main__":
    chatbot()