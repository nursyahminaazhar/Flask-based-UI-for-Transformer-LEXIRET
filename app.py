from flask import Flask, render_template, request
import torch
import torch.nn as nn
import os

# === Vocabulary and tokenizer ===
vocab = [
    "i", "want", "to", "drink", "eat", "stop", "help", "need", "can", "you",
    "me", "please", "<mask>", "<pad>", "pain", "toilet", "hello",
    "thank", "thank you", "yes", "no", "feel", "say", "the", "wants", "hi"
]
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)
max_len = 8
d_model = 64

def tokenize(sentence):
    words = sentence.strip().lower().split()
    return [word2idx.get(word, word2idx["<pad>"]) for word in words]

def encode_and_pad(sentence):
    ids = tokenize(sentence)
    if len(ids) < max_len:
        ids += [word2idx["<pad>"]] * (max_len - len(ids))
    return ids[:max_len]

def get_positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return angle_rads

# === Model definition ===
class ConfidenceTransformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.conf_proj = nn.Linear(1, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, conf):
        seq_len = x.size(1)
        embed = self.embedding(x)
        pos = get_positional_encoding(seq_len, embed.size(-1)).to(x.device)
        embed = embed + pos
        attn_output, _ = self.attn(embed, embed, embed)
        x = self.norm1(embed + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        conf_vec = self.conf_proj(conf).unsqueeze(1)
        x = x + conf_vec
        return self.fc_out(x)

# === Load the trained model ===
model_path = os.path.join("model", "confidence_transformer1.pth")
model = ConfidenceTransformer(vocab_size, d_model)
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
except Exception as e:
    print("❌ Failed to load model:", e)
model.eval()

# === Flask app ===
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        sentence = request.form["input_sentence"]
        confidence_str = request.form.get("confidence", "0.9")  # default confidence 0.9

        # Validate input sentence and <mask> presence
        if "<mask>" not in sentence.lower():
            prediction = "Please include '<mask>' token in the sentence."
            return render_template("index.html", prediction=prediction)

        # Validate confidence input
        try:
            confidence = float(confidence_str)
            if not (0 <= confidence <= 1):
                raise ValueError
        except:
            prediction = "Confidence must be a number between 0 and 1."
            return render_template("index.html", prediction=prediction)

        # Tokenize and pad
        x_ids = encode_and_pad(sentence)
        x_tensor = torch.tensor([x_ids])
        conf_tensor = torch.tensor([[confidence]], dtype=torch.float32)

        # Find mask position index
        tokens = sentence.lower().split()
        try:
            mask_pos = tokens.index("<mask>")
        except ValueError:
            prediction = "No <mask> token found."
            return render_template("index.html", prediction=prediction)

        # Predict
        with torch.no_grad():
            logits = model(x_tensor, conf_tensor)
            pred_idx = torch.argmax(logits[:, mask_pos, :], dim=-1).item()
            pred_word = idx2word.get(pred_idx, "[unknown]")
            user_type = "Normal" if confidence >= 0.7 else "Aphasic"
            prediction = f"Predicted Word: {pred_word} (User Type: {user_type})"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print("❌ Flask app failed to run:", e)

