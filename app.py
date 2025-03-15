from flask import Flask, render_template, request
import requests
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from deep_translator import GoogleTranslator
from bs4 import BeautifulSoup

app = Flask(__name__)

class AIAssistant:
    def __init__(self):
        # Model ve tokenizer'ı yükleyin
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # Padding token'ı ayarla
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()

    def correct_text(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        # Yanıtı daha anlamlı hale getirmek için parametreler üzerinde oynama yapıyoruz
        outputs = self.model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_length=50,  # Cevap uzunluğunu kısıtladık
            temperature=0.6,  # Daha tutarlı ve mantıklı yanıtlar
            top_k=30,  # Daha seçici yanıtlar
            top_p=0.85,  # Nucleus sampling'i sıkı tutalım
            do_sample=True,  # Örnekleme ile yanıtlar üretelim
            pad_token_id=self.tokenizer.eos_token_id  # Padding token'ı ayarlayalım
        )
        
        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text

    def search_duckduckgo(self, query):
        # DuckDuckGo'da arama yapma
        url = f"https://duckduckgo.com/html/?q={query}"
        response = requests.get(url)  # requests kütüphanesini kullanıyoruz
        if response.status_code == 200:
            # HTML içeriğini işle
            soup = BeautifulSoup(response.text, 'html.parser')
            result = soup.find('a', {'class': 'result__a'})  # İlk sonuç bağlantısını bul
            snippet = soup.find('a', {'class': 'result__snippet'})  # Sonuç açıklamasını al
            if result and snippet:
                return f"{result.get_text()} - {snippet.get_text()}"
            elif result:
                return result.get_text()  # Yalnızca başlık metni
            else:
                return "No relevant results found."
        return "No results found."

    def process_input(self, user_input):
        # Eğer "arastir" varsa DuckDuckGo'da arama yap, yoksa metin düzeltme yap
        if "arastir" in user_input.lower():  # Arama terimi "arastir" kontrolü
            query = user_input.lower().replace("arastir", "").strip()  # "arastir" kelimesini çıkar
            print("Searching DuckDuckGo for:", query)
            return self.search_duckduckgo(query)
        else:
            print("Correcting text:", user_input)
            return self.correct_text(user_input)

@app.route("/", methods=["GET", "POST"])
def index():
    assistant = AIAssistant()
    result = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        result = assistant.process_input(user_input)
        # Çeviri yap
        result = GoogleTranslator(source='en', target='tr').translate(result)  # İngilizce'yi Türkçeye çevir
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
