# Next Word Prediction using LSTM

This project is about building a **Next Word Prediction Model** using LSTM (Long Short-Term Memory networks) in TensorFlow/Keras.  


---

## 1. Project Objective
The aim is to create a language model that can:
- Learn patterns of words from a dataset.
- Predict the next word based on a sequence of previous words.
- Generate meaningful text automatically.

---

## 2. Project Workflow

### Step 1: Importing Libraries
We use TensorFlow/Keras for deep learning, NumPy for arrays, and Pickle to save the tokenizer.  

### Step 2: Data Preparation
- Load a text file (`data.txt`).
- Clean the text (remove newlines, special symbols, and convert to lowercase).
- Tokenize the text (convert words into numbers).
- Create training sequences (input words → target next word).

Example:  
If the sentence is *"the quick brown fox jumps"*, the model learns:
- Input: `the quick brown fox` → Output: `jumps`.

### Step 3: Model Architecture
The model uses:
1. **Embedding Layer** – converts words into dense vectors of fixed size.  
2. **Bidirectional LSTM Layer** – reads sequences both forward and backward.  
3. **Dropout Layer** – prevents overfitting.  
4. **Another LSTM Layer** – captures deeper sequence patterns.  
5. **Dense Layer (ReLU)** – learns non-linear features.  
6. **Dense Layer (Softmax)** – outputs probabilities for the next word.

### Step 4: Model Compilation
- Loss: `categorical_crossentropy`
- Optimizer: `Adam` with learning rate = 0.001

### Step 5: Training
- Batch size = 128
- Epochs = 20 (can be adjusted)
- **Callbacks Used**:
  - `ModelCheckpoint`: saves the best model.
  - `EarlyStopping`: stops training if the loss does not improve.

### Step 6: Text Generation
A function is provided to:
- Take a seed text.
- Predict the next `n` words.
- Use *temperature sampling* to control creativity:
  - Low temperature (0.3) → safer, repetitive text.
  - High temperature (1.0+) → more creative, but sometimes random.

---




## 3. How to Run

1. Clone this repository or copy the notebook.
2. Upload a dataset file as `data.txt` in your working directory.
3. Run the notebook step by step on **Google Colab** (recommended).
4. After training, use the `predict_next_words()` function to generate text.

---


---

## 4. Improvements and Notes
- Instead of plain LSTM, you can use **GRU** (faster) or **Transformers** (more powerful).
- Use larger window size (e.g- 10 words) for better accuracy.
- Train for more epochs and on larger datasets for improved results.
- Pre-trained embeddings like **GloVe** or **Word2Vec** can also be used instead of training embeddings from scratch.

---

## 5. Requirements
- Python 3.8+
- TensorFlow/Keras
- NumPy
- Pickle

Install dependencies:
```bash
pip install tensorflow numpy


