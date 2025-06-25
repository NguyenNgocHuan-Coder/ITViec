import streamlit as st
import pickle
import re, regex
import numpy as np
from scipy.sparse import csr_matrix, hstack

# ========== Load mô hình và vectorizer từ .pkl ==========
with open("Streamlit/Sentiment_Models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("Streamlit/Sentiment_Models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("Streamlit/Sentiment_Models/model_lr.pkl", "rb") as f:
    model_lr = pickle.load(f)

with open("Streamlit/Sentiment_Models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ========== Load các dictionary từ file txt ==========
def load_dict_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return dict(line.split('\t') for line in lines if '\t' in line)

def load_list_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

emoji_dict = load_dict_from_txt("Streamlit/Sentiment_Models/emojicon.txt")
teen_dict = load_dict_from_txt("Streamlit/Sentiment_Models/teencode.txt")
wrong_lst = load_list_from_txt("Streamlit/Sentiment_Models/wrong-word.txt")
stopwords_lst = load_list_from_txt("Streamlit/Sentiment_Models/vietnamese-stopwords.txt")
positive_words = load_list_from_txt("Streamlit/Sentiment_Models/positive_VN.txt")
negative_words = load_list_from_txt("Streamlit/Sentiment_Models/negative_VN.txt")
positive_emojis = load_list_from_txt("Streamlit/Sentiment_Models/positive_emoji.txt")
negative_emojis = load_list_from_txt("Streamlit/Sentiment_Models/negative_emoji.txt")

# ========== Tiền xử lý ==========
def covert_unicode(txt):
    return txt.encode('utf-8').decode('utf-8')

def normalize_repeated_characters(text):
    return re.sub(r'(.)\1+', r'\1', text)

def process_text(text):
    document = text.lower().replace("’", '')
    document = regex.sub(r'\.+', ".", document)
    for k, v in emoji_dict.items():
        document = document.replace(k, f" {v} ")
    for k, v in teen_dict.items():
        document = re.sub(rf"\b{k}\b", v, document)
    pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
    words = regex.findall(pattern, document)
    words = [w for w in words if w not in wrong_lst and w not in stopwords_lst]
    return " ".join(words)

def count_sentiment_items(text):
    text = str(text).lower()
    pos_word = sum(1 for word in positive_words if word in text)
    pos_emoji = sum(text.count(emoji) for emoji in positive_emojis)
    neg_word = sum(1 for word in negative_words if word in text)
    neg_emoji = sum(text.count(emoji) for emoji in negative_emojis)
    return pos_word, neg_word, pos_emoji, neg_emoji

# ========== Dự đoán ==========
def predict_sentiment(text_input):
    text = covert_unicode(text_input)
    text = normalize_repeated_characters(text)
    text = process_text(text)

    tfidf_vector = vectorizer.transform([text])
    pos_word, neg_word, pos_emoji, neg_emoji = count_sentiment_items(text_input)
    numeric_features = scaler.transform([[pos_word, neg_word, pos_emoji, neg_emoji]])
    binary_feature = csr_matrix([[1]])
    final_features = hstack([tfidf_vector, csr_matrix(numeric_features), binary_feature])
    y_pred = model_lr.predict(final_features)[0]
    label = le.inverse_transform([y_pred])[0]
    return label

# ========== Giao diện Streamlit ==========
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("📢 Ứng dụng phân tích cảm xúc review công ty")

input_text = st.text_area("✍️ Nhập câu đánh giá của bạn:", height=150)

if st.button("🚀 Dự đoán cảm xúc"):
    if not input_text.strip():
        st.warning("⛔ Vui lòng nhập nội dung review!")
    else:
        with st.spinner("🔍 Đang xử lý..."):
            result = predict_sentiment(input_text)
        st.success(f"✅ Kết quả dự đoán: **{result.upper()}**")
