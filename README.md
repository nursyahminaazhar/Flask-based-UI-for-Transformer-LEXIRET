# Flask-based-UI-for-Transformer-LEXIRET

# Transformer-based Lexical Retrieval (Web-based UI)

This project provides a web-based interface using Flask to predict masked words in a sentence using a Transformer-based model trained for lexical retrieval in aphasic and normal individuals.

## Folder Structure

- `app.py`: Main Flask app.
- `confidence_transformer1.pth`: Trained PyTorch model.
- `templates/index.html`: UI layout.
- `static/style/`: Styling for the interface.

## How to Run

1. **Open Terminal** and navigate to the folder:
   ```bash
   cd "C:\Users\gesture_web_app"

Activate Virtual Environment (if needed):
bash: venv\Scripts\activate

Install Depedencies 
bash: pip install -r requirements.txt

Run The App
bash: python app.py

Open in Browser:
Visit http://127.0.0.1:5000/
