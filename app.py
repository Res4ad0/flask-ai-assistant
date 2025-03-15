from flask import Flask, render_template, request
import requests
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from deep_translator import GoogleTranslator
from bs4 import BeautifulSoup

app = Flask(__name__, static_folder='static', template_folder='templates')

class AIAssistant:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def correct_text(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_length=40,
            temperature=0.5,
            top_k=30,
            top_p=0.85,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text

    def search_duckduckgo(self, query):
        url = f"https://duckduckgo.com/html/?q={query}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            result = soup.find('a', {'class': 'result__a'})
            snippet = soup.find('a', {'class': 'result__snippet'})
            if result and snippet:
                return f"{result.get_text()} - {snippet.get_text()}"
            elif result:
                return result.get_text()
            else:
                return "No relevant results found."
        return "No results found."

    def process_input(self, user_input):
        if "arastir" in user_input.lower():
            query = user_input.lower().replace("arastir", "").strip()
            return self.search_duckduckgo(query)
        else:
            return self.correct_text(user_input)

assistant = AIAssistant()

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        result = assistant.process_input(user_input)
        result = GoogleTranslator(source='en', target='tr').translate(result)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
