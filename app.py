from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from flask import Flask, request, render_template

app = Flask(__name__)

# Load tokenizer and model for cardiffnlp/twitter-roberta-base-sentiment, and pipeline once, for performance
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and display results
@app.route('/analyze', methods=['POST'])
def analyzer():
    # Get text
    text = request.form['text']
    #Perform de analysis
    results = sentiment_pipeline(text)
    
    # Map the labels to meaningful text
    label_mapping = {'LABEL_0': 'Negative', 
                     'LABEL_1': 'Neutral', 
                     'LABEL_2': 'Positive'}
    
    # Get the label and score from the result
    label = label_mapping[results[0]['label']]
    score = results[0]['score']
    
    
    return render_template('result.html', text=text, label=label, score=score)

if __name__ == "__main__":
    app.run(debug=True)
