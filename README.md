# 🔍 AI SEO Meta Generator

An AI-powered SEO tool that extracts text from URLs or uploaded files (TXT, PDF, DOCX), identifies key topics and keywords using NLP, and generates optimized SEO meta titles and descriptions using LLM model.

---

## 🚀 Features

- 📝 Accepts raw text, URLs, or uploaded documents (TXT, PDF, DOCX)
- 🧠 Extracts keywords and topics using spaCy
- 🎯 Generates SEO meta title and description with LLM
- 📦 Exports all outputs as a downloadable CSV
- 🌐 Clean Gradio UI, deployable on Hugging Face Spaces or locally

---

## 🧠 Approach

This tool combines the power of:
- `spaCy` for linguistic keyword and topic extraction
- `LLM` for contextual SEO generation
- `Gradio` for building a beautiful, interactive user interface
- Smart routing of input type: raw text, URL scraping, or file parsing

---

## 📸 Screenshot

![SEO App Screenshot1](https://raw.githubusercontent.com/Divyansh8791/AI-seo-meta-generator/main/screenshot1.PNG)
![SEO App Screenshot2](https://raw.githubusercontent.com/Divyansh8791/AI-seo-meta-generator/main/screenshot2.png)
![SEO App Screenshot3](https://raw.githubusercontent.com/Divyansh8791/AI-seo-meta-generator/main/screenshot3.png)
![SEO App Screenshot4](https://raw.githubusercontent.com/Divyansh8791/AI-seo-meta-generator/main/screenshot4.png)
![SEO App Screenshot5](https://raw.githubusercontent.com/Divyansh8791/AI-seo-meta-generator/main/screenshot5.png)

---

## 🛠️ Setup Instructions

1. **Clone the repo**
```bash
git clone https://github.com/Divyansh8791/AI-seo-meta-generator.git
cd AI-seo-meta-generator
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Set up .env file**
```bash
GOOGLE_API_KEY=your_google_api_key
```
4. **Run the app**
```bash
python app.py
```
---

## 📂 File Structure
```bash
📁 ai-seo-meta-generator/
├── app.py                  # Main Gradio app
├── requirements.txt
├── README.md
└── screenshot1.png
└── screenshot2.png
└── screenshot3.png
└── screenshot4.png
└── screenshot5.png
```
---
🙌 Thank you
Built with ❤️ by Divyansh Dhiman
