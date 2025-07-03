from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

app = Flask(__name__)

# Load your fine-tuned BART model and tokenizer
model_path = "saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def summarize(blog):
    #tokenising input
    inputs = tokenizer(blog, max_length=1024, truncation=True, return_tensors='pt')
    #generating summary
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0,num_beams=4, early_stopping=True)
    #decoding summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    input_text = ""
    original_words = 0
    summary_words = 0
    compression_ratio = "0:1"
    processing_time = 0.0

    if request.method == "POST":
        input_text = request.form["input_text"]
        start = time.time()

        summary = summarize(input_text)
        print(summary)

        end = time.time()
        processing_time = round(end - start, 2)
        original_words = len(input_text.split())
        summary_words = len(summary.split())
        compression_ratio = f"{round(original_words / summary_words, 2)}:1" if summary_words > 0 else "N/A"

    return render_template("index.html",
                           input_text=input_text,
                           summary=summary,
                           original_words=original_words,
                           summary_words=summary_words,
                           compression_ratio=compression_ratio,
                           processing_time=processing_time)


if __name__ == "__main__":
    app.run(debug=True)
