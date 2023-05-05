import re
import pickle
import string
import numpy as np
import streamlit as st
from os.path import join
from keras.models import load_model
from keras.utils.data_utils import pad_sequences

MAX_SEQUENCE_LENGTH = 1000

class_dict = {1: "Positive review", 0: "Negative review"}

st.title("Sentiment Analysis App")


def remove_digits_ifany(row):
    new_row = ""
    match = re.search(r"\s\d+\s", row)
    if match:
        new_row += row.replace(match.group(0), "")
    else:
        new_row += row
    return new_row


def expand_shorts(row):
    row = re.sub(r"won't", "will not", row)
    row = re.sub(r"can\'t", "can not", row)
    row = re.sub(r"n\'t", " not", row)
    row = re.sub(r"\'re", " are", row)
    row = re.sub(r"\'s", " is", row)
    row = re.sub(r"\'d", " would", row)
    row = re.sub(r"\'ll", " will", row)
    row = re.sub(r"\'t", " not", row)
    row = re.sub(r"\'ve", " have", row)
    row = re.sub(r"\'m", " am", row)
    return row


def basic_cleanup(row):
    row = re.sub(r"http\S+", "", row)
    row = re.sub("\S*\d\S*", "", row).strip()
    row = re.sub("[^A-Za-z]+", " ", row)
    row = " ".join(word.lower() for word in row.split())
    words = row.split(" ")
    row = " ".join([word for word in words if len(word) < 15 or len(word) > 2])
    return row


def isalpha(word):
    mychars = string.ascii_lowercase + string.ascii_uppercase + "_"
    flag = True
    for char in word:
        if char not in mychars:
            flag = False
            break
    return flag


def replace_non_alpha(row):
    words = row.split(" ")
    row = " ".join([word for word in words if isalpha(word)])
    return row


def load_tokenizer(path):
    with open(path, "rb") as fh:
        tokenizer = pickle.load(fh)
    return tokenizer


def load_trained_model():
    best_model_path = "best_model_1.h5"
    model = load_model(join("static", best_model_path))
    return model


def predict_class(text_in):
    model = load_trained_model()
    tokenizer_1 = load_tokenizer(join("static", "tokenizer_1.pickle"))
    tokenizer_2 = load_tokenizer(join("static", "tokenizer_2.pickle"))
    inp_embd_1 = tokenizer_1.texts_to_sequences(text_in)
    inp_embd_2 = tokenizer_2.texts_to_sequences(text_in)

    inp_seq_1 = pad_sequences(
        inp_embd_1,
        maxlen=MAX_SEQUENCE_LENGTH,
        dtype="uint16",
        padding="pre",
        truncating="pre",
        value=0,
    )
    inp_seq_2 = pad_sequences(
        inp_embd_2,
        maxlen=MAX_SEQUENCE_LENGTH,
        dtype="uint16",
        padding="pre",
        truncating="pre",
        value=0,
    )

    y_pred_probs = model.predict([inp_seq_1, inp_seq_2])
    y_pred = np.argmax(y_pred_probs, axis=1)
    return y_pred


text_in = st.text_input("Type review text here", "Wonderful product, I really love it")
text_in = remove_digits_ifany(text_in)
text_in = expand_shorts(text_in)
text_in = basic_cleanup(text_in)
text_in = replace_non_alpha(text_in)


if st.button("Predict") and text_in:
    key = predict_class([text_in])[0]
    st.text("Output:")
    st.text(class_dict[key])
