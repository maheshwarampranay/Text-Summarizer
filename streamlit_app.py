import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

# Load your fine-tuned BART model and tokenizer
@st.cache_resource
def load_model():
    model_path = "saved_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

st.title("Text Summarizer - using fine-tuned BART")
st.markdown("Enter a long article or blog post to get a concise summary.")

input_text = st.text_area("✏️ Input Text", height=300)

if st.button("Generate Summary"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        start_time = time.time()

        # Tokenize input and generate summary
        inputs = tokenizer(input_text, max_length=1024, truncation=True, return_tensors="pt")
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        end_time = time.time()

        # Display results
        original_words = len(input_text.split())
        summary_words = len(summary.split())
        compression_ratio = f"{round(original_words / summary_words, 2)}:1" if summary_words > 0 else "N/A"
        processing_time = round(end_time - start_time, 2)

        st.subheader("📝 Summary")
        st.success(summary)

        st.markdown("---")
        st.markdown(f"**📊 Original Words:** {original_words}")
        st.markdown(f"**✂️ Summary Words:** {summary_words}")
        st.markdown(f"**🔻 Compression Ratio:** {compression_ratio}")
        st.markdown(f"**⏱️ Time Taken:** {processing_time} seconds")
