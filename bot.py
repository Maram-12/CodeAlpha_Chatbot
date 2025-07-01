import spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FAQs about internships
faqs = [
    {
        "question": "What is an internship?",
        "answer": "An internship is a temporary work experience offered by companies or organizations to students or recent graduates, allowing them to gain practical skills and industry knowledge in their field of study."
    },
    {
        "question": "Are internships paid or unpaid?",
        "answer": "Internships can be either paid or unpaid. Paid internships provide compensation (hourly wage or stipend), while unpaid internships offer experience and sometimes academic credit instead of payment."
    },
    {
        "question": "How long do internships typically last?",
        "answer": "Most internships last between 2-6 months, with summer internships typically being 8-12 weeks. Some part-time internships during academic terms may last longer (3-6 months)."
    },
    {
        "question": "When should I apply for summer internships?",
        "answer": "For summer internships, applications typically open 6-9 months in advance. The best time to apply is usually between September and February for the following summer."
    },
    {
        "question": "What documents do I need for an internship application?",
        "answer": "Common requirements include: Resume/CV, Cover letter, Academic transcripts, Letters of recommendation, and Portfolio (for creative fields)."
    },
    {
        "question": "Can international students apply for internships?",
        "answer": "Yes, but they may need additional documentation like a valid student visa (F-1 in the US) and possibly CPT/OPT authorization. Some companies may have restrictions."
    },
    {
        "question": "How can I make my internship application stand out?",
        "answer": "Highlight relevant coursework, projects, and skills. Tailor your resume for each position, include measurable achievements, and demonstrate enthusiasm for the specific company/role."
    },
    {
        "question": "What should I expect from my first day as an intern?",
        "answer": "Typically includes orientation, meeting your team, learning about company policies, setting up your workspace, and receiving initial assignments. Dress professionally and come prepared with questions."
    },
    {
        "question": "Can an internship lead to a full-time job?",
        "answer": "Yes, many companies use internships as a recruitment pipeline. About 50-60% of interns receive full-time job offers from their internship employers according to NACE data."
    },
    {
        "question": "What's the difference between an internship and a co-op?",
        "answer": "Co-ops are typically longer (3-12 months), more structured, often paid, and may alternate with academic terms. Internships are usually shorter and may not be as integrated with academic programs."
    }
]

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Text preprocessing
def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

# Prepare FAQ data
faq_questions = [q["question"] for q in faqs]
preprocessed_questions = [preprocess(q) for q in faq_questions]

# Vectorize FAQs
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)

# Similarity matching
def get_most_similar(user_query):
    processed_query = preprocess(user_query)
    query_vec = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)
    most_similar_idx = similarities.argmax()
    return faqs[most_similar_idx]["answer"]

# Chatbot logic
def chatbot_response(user_input):
    if user_input.lower() in ["hi", "hello", "hey"]:
        return "Hello! How can I help you today?"
    elif user_input.lower() in ["bye", "goodbye"]:
        return "Goodbye! Have a great day!"
    else:
        try:
            return get_most_similar(user_input)
        except:
            return "I'm sorry, I don't understand that question. Could you rephrase it?"

# Simple CLI
print("FAQ Chatbot: Type 'quit' or 'q' to exit")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'q']:
        break
    response = chatbot_response(user_input)
    print("Bot:", response)


