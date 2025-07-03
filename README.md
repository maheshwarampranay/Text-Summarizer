Here’s a clean, well-structured `README.md` for your GitHub repository:

---

# Text Summarizer using Fine-Tuned BART

This project demonstrates a complete pipeline for abstractive text summarization using a fine-tuned BART transformer model. The model is trained on a custom dataset (other than CNN/DailyMail) and deployed via a Streamlit application. The entire process includes data preparation, model training, Hugging Face model upload, and deployment.

---

## 📌 Project Overview

Text summarization is the task of shortening long pieces of text into concise summaries while preserving key information and overall meaning. This project uses the **BART** (Bidirectional and Auto-Regressive Transformers) architecture by Facebook AI, which is well-suited for sequence-to-sequence tasks like summarization.

---

## 🧠 About BART

BART is a denoising autoencoder for pretraining sequence-to-sequence models. It combines the bidirectional encoder of BERT and the autoregressive decoder of GPT. BART has shown strong performance on text generation tasks, especially abstractive summarization.

In this project:

* We start with a pretrained BART model.
* Fine-tune it on a custom dataset to adapt it for summarizing domain-specific or stylistically different content from the original CNN/DailyMail dataset.
* Use Hugging Face’s `transformers` and `datasets` libraries for training and integration.

---

## 📁 Repository Structure

```plaintext
.
├── Text_Summarization.ipynb     # Training pipeline: data loading, model fine-tuning, saving
├── streamlit_app.py             # Deployment app using Streamlit
├── requirements.txt             # Dependencies for running the project
└── README.md                    # Project documentation
```
-- Uploaded the model to hugging face and used it via hugging face hub for deployed version.
---

## 🔧 Training Pipeline

The training notebook (`Text_Summarization.ipynb`) walks through:

* Dataset loading and preprocessing
* Tokenization using BART tokenizer
* Model training with `Trainer` from Hugging Face
* Evaluation and saving the fine-tuned model
* Uploading the trained model to Hugging Face model hub

The model was fine-tuned on a custom summarization dataset, improving its performance on different content compared to the CNN/DailyMail dataset.

---

## 🚀 Deployment

The `streamlit_app.py` file enables users to interact with the fine-tuned BART model via a web interface.

Features:

* Simple input box for pasting article or blog text
* Generates summaries using the loaded model
* Displays:

  * Final summary
  * Original word count
  * Summary word count
  * Compression ratio
  * Time taken for summarization

The model is loaded from the Hugging Face model hub or locally (based on setup).

---

## 🛠️ How to use it ?

Access the deployed link in the description of this repository.

## 📦 Model Access

The model we developed:
**(https://huggingface.co/pranaymaheshwaram/Text-summarizaer-fine-tuned-BART)**


---

## 📚 Tech Stack

* Python
* Hugging Face Transformers
* PyTorch
* Streamlit

---
![Screenshot 2025-07-03 121120](https://github.com/user-attachments/assets/90560842-41d3-4bb2-b314-e96c78860667)
![Screenshot 2025-07-03 121207](https://github.com/user-attachments/assets/c815d391-18d4-4b55-8159-52c66a1adb01)
![Screenshot 2025-07-03 121136](https://github.com/user-attachments/assets/44ac27c7-c899-4213-8442-b5b28ccae6ce)

## 📄 License

This project is released under the [MIT License](LICENSE).

---

