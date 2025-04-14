import streamlit as st
import torch
import pickle
from model import Transformer
from vocab import Vocabulary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SRC_PAD_IDX = 0
TGT_PAD_IDX = 0
MAX_LEN = 50

# Load vocabularies
with open("en_vocab.pkl", "rb") as f:
    en_vocab = pickle.load(f)

with open("hi_vocab.pkl", "rb") as f:
    hi_vocab = pickle.load(f)

# Load model (must match training config: max_len=50)
model = Transformer(
    src_vocab_size=len(en_vocab),
    tgt_vocab_size=len(hi_vocab),
    max_len=MAX_LEN
).to(DEVICE)

model.load_state_dict(torch.load("transformer_translation_final.pth", map_location=DEVICE))
model.eval()

# Utility functions
def encode_sentence(sentence, vocab, max_len=MAX_LEN):
    tokens = [vocab["<sos>"]] + vocab.numericalize(sentence) + [vocab["<eos>"]]
    tokens = tokens[:max_len]
    tokens += [vocab["<pad>"]] * (max_len - len(tokens))
    return tokens

def translate_sentence(model, sentence, en_vocab, hi_vocab, max_len=MAX_LEN):
    model.eval()
    tokens = encode_sentence(sentence, en_vocab, max_len)
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

    tgt_tokens = [hi_vocab["<sos>"]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, SRC_PAD_IDX, TGT_PAD_IDX)
        next_token = output[0, -1].argmax().item()
        tgt_tokens.append(next_token)
        if next_token == hi_vocab["<eos>"]:
            break

    translated = [hi_vocab.itos[idx] for idx in tgt_tokens[1:-1]]
    return ' '.join(translated)

# Streamlit app
st.title("English to Hindi Translation 🇮🇳")
st.write("Enter a sentence in English and get its Hindi translation using a Transformer model!")

user_input = st.text_input("Enter English sentence:")

if user_input:
    translation = translate_sentence(model, user_input, en_vocab, hi_vocab)
    st.subheader("Hindi Translation:")
    st.success(translation)
