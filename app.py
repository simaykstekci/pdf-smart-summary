import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

st.set_page_config(page_title="PDF Smart Summary", page_icon="📄", layout="centered")
st.title("📄 PDF Smart Summary")
st.write("PDF yükle → Akıllı özet otomatik oluşturulsun!")

def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except:
        return ""

def summarize_text(text, n_sentences=4):
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) <= n_sentences:
        return text  # kısa metni özetlemeden ver
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)
    scores = np.mean(tfidf_matrix.toarray(), axis=1)
    top_sentence_indices = np.argsort(scores)[-n_sentences:]
    summary = ". ".join([sentences[i] for i in sorted(top_sentence_indices)])
    return summary

pdf_file = st.file_uploader("PDF dosyanı yükle", type=["pdf"])

if pdf_file:
    with st.spinner("PDF okunuyor ve özetleniyor..."):
        text = extract_text_from_pdf(pdf_file)
        if not text:
            st.error("PDF'den metin çıkarılamadı. Lütfen farklı bir PDF deneyin.")
        else:
            st.subheader("📌 Orijinal Metin")
            st.text_area("Orijinal Metin", text, height=300)
            
            st.subheader("✨ Smart Summary")
            summary = summarize_text(text)
            st.text_area("Özet", summary, height=200)
