# 🤖 AI Learning Companion

> An intelligent, multimodal learning assistant that combines computer vision, NLP, and OCR to help you understand any image-based content — whether it's a diagram, a page of text, or a mixed document.

---

## 📌 Overview

**AI Learning Companion** is a Streamlit web application that lets you upload any image and ask questions about it in plain English. The app automatically detects objects in the image, extracts text via OCR, classifies the content type, understands your intent, and generates a tailored AI response — an explanation, a summary, or a set of quiz questions.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Object Detection** | Detects and annotates objects in the uploaded image using YOLOv8 |
| 📄 **OCR Text Extraction** | Extracts readable text from images using Tesseract OCR |
| 🧠 **Content Classification** | Custom-trained CNN classifies image content as *Text*, *Diagram*, or *Mixed* |
| 🎯 **Intent Detection** | Custom LSTM model classifies your query intent as *Explain*, *Summary*, or *Quiz* |
| 💬 **Adaptive AI Response** | Generates a contextual response — explanation, summary, or practice questions — based on detected intent |

---

## 🗂️ Project Structure

```
AI-Learn/
├── app.py                          # Main Streamlit application
├── cnn_model.ipynb                 # CNN training notebook (content classification)
├── lstm_model.ipynb                # LSTM training notebook (intent classification)
└── models/
    ├── cnn_best.h5                 # Trained CNN model
    ├── advanced_intent_lstm.h5     # Trained LSTM model
    ├── tokenizer.pkl               # Tokenizer for intent model
    └── max_len.pkl                 # Sequence max length for padding
```

---

## 🧩 Architecture

```
User uploads image + types query
        │
        ├──► YOLOv8 (Object Detection) ──► Annotated image + object labels
        │
        ├──► Tesseract OCR ──────────────► Extracted text
        │
        ├──► CNN (128×128 input) ────────► Content type: Text / Diagram / Mixed
        │
        └──► LSTM + Tokenizer ───────────► Intent: Explain / Summary / Quiz
                                                    │
                                                    ▼
                                         Adaptive AI Response
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system

> **Windows users:** After installing Tesseract, update the path in `app.py`:
> ```python
> pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
> ```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anuragpp77/AI-Learn.git
   cd AI-Learn
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit tensorflow ultralytics opencv-python pytesseract numpy
   ```

3. **Ensure model files are present** under the `models/` directory:
   - `cnn_best.h5`
   - `advanced_intent_lstm.h5`
   - `tokenizer.pkl`
   - `max_len.pkl`

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 🎮 Usage

1. Open the app in your browser (defaults to `http://localhost:8501`)
2. **Upload an image** — a textbook page, a diagram, a chart, a worksheet, etc.
3. **Type a question** in plain English, such as:
   - `"explain this"`
   - `"summarise the content"`
   - `"quiz me"`
4. The app will return:
   - An annotated image with detected objects
   - Extracted text from the image
   - Detected content type and intent with confidence scores
   - A tailored AI response

---

## 🧠 Models

### CNN — Content Type Classifier
- **Input:** 128×128 RGB image
- **Output:** `Text` | `Diagram` | `Mixed`
- **Framework:** TensorFlow / Keras
- **Notebook:** `cnn_model.ipynb`

### LSTM — Intent Classifier
- **Input:** Tokenised and padded user query
- **Output:** `Explain` | `Summary` | `Quiz`
- **Framework:** TensorFlow / Keras
- **Notebook:** `lstm_model.ipynb`

### YOLOv8 — Object Detector
- **Model:** `yolov8m.pt` (pretrained on COCO)
- **Library:** Ultralytics

---

## 🛠️ Tech Stack

- **Frontend / UI:** Streamlit
- **Deep Learning:** TensorFlow, Keras
- **Object Detection:** Ultralytics YOLOv8
- **OCR:** Tesseract via `pytesseract`
- **Image Processing:** OpenCV, NumPy

---

## 📸 Example Queries

| Query | Detected Intent | Response |
|---|---|---|
| `"explain this"` | Explain | Detailed breakdown of image content |
| `"summarise"` | Summary | Concise 300-character summary of extracted text |
| `"quiz me"` | Quiz | Up to 3 practice questions based on extracted text |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Anurag** — AI/ML Engineer  
[GitHub](https://github.com/anuragpp77)
