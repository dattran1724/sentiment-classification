import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

# ===============================
# CONFIG
# ===============================
MODEL_NAME = "bert-base-uncased"
CHECKPOINT_PATH = "D:\\2025-2026\\NPL_CK\\models\\best_model_checkpoint_mBERT_unfreeze.pth"
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 33

# ===============================
# MODEL DEFINITION
# ===============================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.u = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        u_it = torch.tanh(self.W(x))                 # (batch, seq_len, hidden_size)
        a_it = self.u(u_it).squeeze(-1)             # (batch, seq_len)
        a_it = F.softmax(a_it, dim=1).unsqueeze(-1) # (batch, seq_len, 1)
        a_it = self.dropout(a_it)
        weighted_input = x * a_it
        return weighted_input.sum(dim=1)            # (batch, hidden_size)

class BERT_BiLSTM_Attention_MultiTask(nn.Module):
    def __init__(self, bert_model_name, lstm_hidden=64, 
                 num_category_classes=NUM_CLASSES, num_polarity_classes=3, dropout=0.3):
        super(BERT_BiLSTM_Attention_MultiTask, self).__init__()
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        
        # BiLSTM 2 layers
        self.lstm = nn.LSTM(hidden_size, lstm_hidden, 
                            batch_first=True, bidirectional=True, 
                            dropout=dropout, num_layers=2)
        
        # Attention
        self.attention = Attention(lstm_hidden*2)
        
        self.batch_norm = nn.BatchNorm1d(lstm_hidden*2)
        
        # Shared classifier layers
        self.fc1 = nn.Linear(lstm_hidden*2, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        
        # Multi-task heads
        self.fc_category = nn.Linear(64, num_category_classes)
        self.fc_polarity = nn.Linear(64, num_polarity_classes)
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_output.last_hidden_state               # (batch, seq_len, hidden_size)
        x, _ = self.lstm(x)                             # BiLSTM
        x = self.attention(x)                           # Attention
        x = self.batch_norm(x)                          # batch_norm
        
        # Shared MLP
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Multi-task outputs
        out_category = self.fc_category(x)
        out_polarity = self.fc_polarity(x)
        return out_category, out_polarity

# ===============================
# TEXT CLEANING
# ===============================
def clean_text(text: str) -> str:
    if isinstance(text, list) and len(text) > 0:
        text = text[0]
    elif not isinstance(text, str):
        text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    emoji_pattern = re.compile("["  
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002702-\U000027B0"  
        u"\U000024C2-\U0001F251"  
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text.strip())
    return text

# ===============================
# LABEL ENCODERS
# ===============================
df_aug = pd.read_csv(r"D:\2025-2026\NPL_CK\New_data\csv\df_aug_clean_v2_no_chinese.csv")
le_cat = LabelEncoder()
le_pol = LabelEncoder()

le_cat.fit(df_aug['category'].astype(str).tolist())
le_pol.fit(df_aug['polarity'].astype(str).tolist())

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_multitask(text, model, tokenizer, device, le_cat, le_pol):
    model.eval()
    text = clean_text(text)
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs_cat, outputs_pol = model(input_ids, attention_mask)
        
        probs_cat = F.softmax(outputs_cat, dim=1).cpu().numpy()[0]
        pred_cat = np.argmax(probs_cat)
        label_cat = le_cat.inverse_transform([pred_cat])[0]
        
        probs_pol = F.softmax(outputs_pol, dim=1).cpu().numpy()[0]
        pred_pol = np.argmax(probs_pol)
        label_pol = le_pol.inverse_transform([pred_pol])[0]

    return {
        'category': label_cat,
        'category_confidence': float(probs_cat[pred_cat]),
        'category_probs': {le_cat.classes_[i]: float(probs_cat[i]) for i in range(len(probs_cat))},
        'polarity': label_pol,
        'polarity_confidence': float(probs_pol[pred_pol]),
        'polarity_probs': {le_pol.classes_[i]: float(probs_pol[i]) for i in range(len(probs_pol))}
    }


# ===============================
# LOAD MODEL & TOKENIZER
# ===============================
from torch.optim import AdamW
def build_optimizer(model):
    return AdamW([
        {"params": model.bert.embeddings.parameters(), "lr": 5e-6},
        {"params": model.bert.encoder.parameters(), "lr": 2e-5},
        {"params": model.lstm.parameters(), "lr": 1e-3},
        {"params": model.fc1.parameters(), "lr": 1e-3},
        {"params": model.fc2.parameters(), "lr": 1e-3},
        {"params": model.fc_category.parameters(), "lr": 1e-3},
        {"params": model.fc_polarity.parameters(), "lr": 1e-3},
    ], weight_decay=0.01)
    


def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint"""
    try:
        checkpoint = torch.load(filename, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        le_cat.classes_ = np.array(checkpoint['le_cat_classes'])
        le_pol.classes_ = np.array(checkpoint['le_pol_classes'])
        print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
        return model, optimizer, epoch, loss
    except FileNotFoundError:
        print(f"Checkpoint file {filename} not found")
        return model, optimizer, 0, float('inf')

@st.cache_resource
def load_model_for_inference():
    tokenizer = AutoTokenizer.from_pretrained(r"D:\2025-2026\NPL_CK\my_finetuned_tokenizer")
    model = BERT_BiLSTM_Attention_MultiTask(r"D:\2025-2026\NPL_CK\my_finetuned_bert").to(DEVICE)
    optimizer = build_optimizer(model)
    # Load best checkpoint
    model, _ , _, _ = load_checkpoint(model, optimizer, CHECKPOINT_PATH )
    return model, tokenizer

model, tokenizer = load_model_for_inference()

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(
    page_title="Aspect-based Sentiment Analysis",
    layout="wide"
)

st.title("Aspect-based Sentiment Analysis")
st.markdown("**Multi-task model: Aspect Category + Sentiment Polarity**")

text_input = st.text_area(
    "Enter your review text:",
    height=150,
    placeholder="Example: The battery life is amazing but the screen is terrible"
)

if st.button("Analyze Sentiment"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_multitask(
                text_input,
                model,
                tokenizer,
                DEVICE,
                le_cat,
                le_pol
            )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Aspect Category")
            st.success(result["category"])
            st.metric("Confidence", f"{result['category_confidence']:.2%}")
            df_cat = pd.DataFrame(result["category_probs"].items(),
                                  columns=["Category", "Probability"]).sort_values("Probability", ascending=False)
            st.bar_chart(df_cat.set_index("Category"))

        with col2:
            st.subheader("Sentiment Polarity")
            polarity_color = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}
            st.success(f"{polarity_color.get(result['polarity'], '')} {result['polarity']}")
            st.metric("Confidence", f"{result['polarity_confidence']:.2%}")
            df_pol = pd.DataFrame(result["polarity_probs"].items(),
                                  columns=["Polarity", "Probability"]).sort_values("Probability", ascending=False)
            st.bar_chart(df_pol.set_index("Polarity"))

st.markdown("---")
st.caption("Aspect-based Sentiment Analysis | BERT + BiLSTM + Attention (Multi-task)")

