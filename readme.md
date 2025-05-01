# 🎬 Predicting Movie Review Sentiments with Machine Learning: An IMDb Case Study

This project uses machine learning and deep learning techniques to predict the sentiment of IMDb movie reviews—positive or negative—based on their textual content. We implemented an LSTM-based neural network to handle sequential text data and achieved strong predictive performance.

---

## 📌 Problem Statement

To classify IMDb movie reviews as either **positive** or **negative** using Natural Language Processing (NLP) and a deep learning model. This is a common real-world use case in customer feedback analysis, opinion mining, and recommendation systems.

---

## Project Structure

IMDb-Sentiment-Analysis/
├── data/                # Dataset and any preprocessing
├── notebooks/           # Jupyter notebooks for EDA and modeling
├── src/                 # Core Python scripts (if modularized)
├── images/              # Word clouds, graphs, confusion matrix
├── requirements.txt     # Python dependencies
├── README.md            # Project summary


## 📊 Dataset

- **Source**: [IMDb 50K Movie Reviews - Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 reviews (25K positive, 25K negative)
- **Structure**: Each row contains a movie review and its sentiment label.

---

## 🔍 Data Preprocessing & Exploration

- Converted sentiment labels to binary (positive → 1, negative → 0)
- Stratified split: 80% train, 10% validation, 10% test
- Generated word clouds to visualize commonly used terms in both sentiment classes

---

## Running the IMDB_Sentiment_Analysis.ipynb on google colab

- First go to File -> upload Notebook, upload the IMDB_Sentiment_Analysis.ipynb file
- Go to Runtime -> Run all command
- All the cells will run sequentially generating the desired output

## Running the Code Locally

- First create python virtual environment and activate
""
  python -m venv venv
  venv/Scripts/Activate //for windows
  source venv/bin/activate //for macbook

""
- Run the requirements.txt file to install all dependencies
""
  pip install -r requirements.txt
""

- Finally run the local.py file
""
  python local.py
""

## 🧠 Model Architecture

Built with TensorFlow & Keras:

model = Sequential()
model.add(Embedding(input_dim=5001, output_dim=128))
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=1, activation='sigmoid'))

Embedding Layer: Maps words to dense vectors

LSTM Layer: Captures sequential dependencies

Dense Output: Sigmoid activation to predict binary sentiment

⚙️ Training Details
Loss Function: Binary Crossentropy

Optimizer: Adam

Batch Size: 128

Epochs: 9 (Early stopping used to avoid overfitting)

📈 Performance
Validation Accuracy: 88.04%

Test Accuracy: ~87%

Confusion Matrix:

True Positives: 2,245

True Negatives: 2,159

False Positives: 341

False Negatives: 255

💡 Sample Predictions
"This movie was amazing!" → ✅ Positive (Confidence: 0.97)

"This movie was average for others, but I really loved it!" → ✅ Positive (Confidence: 0.86)

"disgusting" → ❌ Negative (Confidence: 0.22)

🧠 Key Takeaways
Preprocessing and clean, balanced datasets boost model performance

LSTM models work well for context-based sentiment analysis

Confidence scores help understand edge case predictions

⚠️ Limitations
Struggles with sarcasm and mixed sentiments

May not generalize well to other domains (e.g., product reviews)

🛠️ Technologies Used
Python

Pandas, NumPy

TensorFlow & Keras

Matplotlib, Seaborn

NLTK / Text Preprocessing Tools

## License
This project is for academic purposes.

## Contact
For any questions or collaboration opportunities, feel free to reach out via LinkedIn (linked above).

