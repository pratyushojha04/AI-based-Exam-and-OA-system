from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from torch.nn.functional import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from datetime import datetime
import os
import random

app = Flask(__name__)

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load the questions and answers dataset
questions_file_path = 'machine_learning_questions_with_answers.csv'

questions_df = pd.read_csv(questions_file_path)

# Function to extract features from answers
def extract_features(answer):
    inputs = tokenizer(answer, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state.squeeze(0)
    attention_mask = inputs['attention_mask'].squeeze(0)
    valid_token_embeddings = token_embeddings[attention_mask.bool()]
    mean_embedding = torch.mean(valid_token_embeddings, dim=0)
    return mean_embedding

# Function to calculate cosine similarity
def calculate_similarity(vector1, vector2):
    vector1 = vector1.unsqueeze(0) if vector1.ndim == 1 else vector1
    vector2 = vector2.unsqueeze(0) if vector2.ndim == 1 else vector2
    similarity = cosine_similarity(vector1, vector2)
    return similarity.item()

# Initialize DataFrame for applicants
df_file_path = 'applicants.csv'
if os.path.exists(df_file_path):
    df = pd.read_csv(df_file_path)
else:
    df = pd.DataFrame(columns=['Name', 'Q1_vector', 'Q2_vector', 'Q3_vector', 'Q4_vector', 'Q5_vector', 'total_score', 'rank', 'submission_time'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/questions', methods=['GET', 'POST'])
def questions():
    if request.method == 'POST':
        name = request.form['name']
        answers = [
            request.form['answer1'],
            request.form['answer2'],
            request.form['answer3'],
            request.form['answer4'],
            request.form['answer5']
        ]
        
        total_score = 0
        scores = []
        
        # Loop through the answers and calculate scores
        for i, answer in enumerate(answers):
            answer_vector = extract_features(answer)
            reference_vector = extract_features(questions_df['Answer'][i])  # Fetch the correct answer vector
            similarity = calculate_similarity(reference_vector, answer_vector)
            scores.append(similarity)
            total_score += similarity
        
        # Add the current submission time
        submission_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Append the applicant's data
        df.loc[len(df)] = [name, *scores, total_score, None, submission_time]
        df['rank'] = df['total_score'].rank(method='first', ascending=False).astype(int)
        
        # Save the updated DataFrame to CSV
        df.to_csv(df_file_path, index=False)
        
        return redirect(url_for('result'))

    # Select 5 random questions
    random_questions = questions_df.sample(n=5).reset_index(drop=True)
    return render_template('questions.html', questions=random_questions)

@app.route('/result')
def result():
    sorted_df = df.sort_values(by='rank', ascending=True)
    return render_template('result.html', data=sorted_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
