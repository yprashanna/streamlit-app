import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json

# Load the fine-tuned model
model = SentenceTransformer('fine-tuned-model')

# Load the JSON file
with open('scholarships.json', 'r') as f:
    data = json.load(f)

@st.cache
def get_scholarship_data():
    return data

@st.cache(allow_output_mutation=True)
def load_model():
    return model

data = get_scholarship_data()
model = load_model()

def find_answer(question):
    questions = []
    answers = []
    for section in data.values():
        if isinstance(section, list):
            for item in section:
                if isinstance(item, dict) and 'question' in item and 'answer' in item:
                    questions.append(item['question'])
                    answers.append(item['answer'])

    if not questions:
        return "Sorry, I don't have enough data to answer that question."

    # Encode the question and the list of questions
    question_embedding = model.encode(question, convert_to_tensor=True)
    questions_embeddings = model.encode(questions, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(question_embedding, questions_embeddings)

    # Find the index of the highest score
    best_index = cosine_scores.argmax()
    best_score = cosine_scores[0][best_index].item()

    if best_score < 0.5:  # Threshold can be adjusted
        return "Sorry, I couldn't find a relevant answer."

    return answers[best_index]

st.title("Scholarship FAQ Bot")

user_question = st.text_input("Ask a question about the scholarship:")
if user_question:
    answer = find_answer(user_question)
    st.write(f"Answer: {answer}")
