import streamlit as st
import re, regex
import numpy as np
from underthesea import word_tokenize, pos_tag, sent_tokenize
from scipy.sparse import csr_matrix, hstack

# Giả định bạn đã huấn luyện sẵn các biến sau:
# model_lr, vectorizer, scaler, le
# Các dict, list: emoji_dict, teen_dict, wrong_lst, stopwords_lst
# Các từ cảm xúc: positive_words, negative_words, positive_emojis, negative_emojis
# Và các hàm: covert_unicode, normalize_repeated_characters, translate_words, ...

# ========== Tiền xử lý giống pipeline ==========
def process_text(text, emoji_dict, teen_dict, wrong_lst):
    document = text.lower().replace("’", '')
    document = regex.sub(r'\.+', ".", document)
    new_sentence = ''
    for sentence in sent_tokenize(document):
        sentence = ''.join(emoji_dict.get(char, char) for char in sentence)
        sentence = ' '.join(teen_dict.get(word, word) for word in sentence.split())
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern, sentence))
        sentence = ' '.join(word for word in sentence.split() if word not in wrong_lst)
        new_sentence += sentence + '. '
    return regex.sub(r'\s+', ' ', new_sentence).strip()

def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.', '')
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        words = word_tokenize(sentence, format='text')
        tagged = pos_tag(words)
        filtered = ' '.join(w for w, t in tagged if t in lst_word_type)
        new_document += filtered + ' '
    return regex.sub(r'\s+', ' ', new_document).strip()

def remove_stopword(text, stopwords):
    return regex.sub(r'\s+', ' ', ' '.join(word for word in text.split() if word not in stopwords)).strip()

def count_sentiment_items(text, words_list, emojis_list):
    text = str(text).lower()
    word_count = sum(1 for word in words_list if word in text)
    emoji_count = sum(text.count(emoji) for emoji in emojis_list)
    return word_count, emoji_count

# ========== Hàm dự đoán ==========
def predict_sentiment(text_input):
    text = covert_unicode(text_input)
    text = normalize_repeated_characters(text)
    text = translate_words(text)
    text = process_text(text, emoji_dict, teen_dict, wrong_lst)
    text = process_postag_thesea(text)
    text = remove_stopword(text, stopwords_lst)

    tfidf_vector = vectorizer.transform([text])
    pos_word, pos_emoji = count_sentiment_items(text_input, positive_words, positive_emojis)
    neg_word, neg_emoji = count_sentiment_items(text_input, negative_words, negative_emojis)
    numeric_features = scaler.transform([[pos_word, neg_word, pos_emoji, neg_emoji]])
    binary_feature = csr_matrix([[1]])  # Recommend = 1 mặc định

    final_features = hstack([tfidf_vector, csr_matrix(numeric_features), binary_feature])
    prediction = model_lr.predict(final_features)[0]
    label = le.inverse_transform([prediction])[0]
    return label

# ========== Giao diện Streamlit ==========
st.title("📊 Ứng dụng phân tích cảm xúc review công ty")

review_input = st.text_area("✍️ Nhập vào câu đánh giá:", height=150)

if st.button("Dự đoán cảm xúc"):
    if review_input.strip() == "":
        st.warning("Vui lòng nhập câu đánh giá!")
    else:
        with st.spinner("Đang xử lý..."):
            result = predict_sentiment(review_input)
        st.success(f"✅ Kết quả dự đoán: **{result.upper()}**")

