# 🛡️ Federated Learning for Credit Card Fraud Detection

This project implements a privacy-preserving **Federated Learning** system to detect fraudulent credit card transactions using a deep learning model. It simulates multiple client environments, applies federated averaging, and introduces differential privacy for enhanced security.

---

## 🔍 Project Overview

- **Problem**: Credit card fraud detection using decentralized data.
- **Solution**: Federated Learning approach with Deep Neural Networks.
- **Privacy**: Adds Differential Privacy using Laplacian Noise.
- **Clients**: 5 simulated clients with stratified data distribution.
- **Evaluation**: Accuracy, Precision, Recall, AUC, Confusion Matrix, ROC Curve.

---

## 🧠 Model Architecture

The model is a deep neural network built with Keras:

- `Dense(64, activation='relu')`  
- `Dropout(0.3)`
- `Dense(32, activation='relu')`  
- `Dropout(0.3)`
- `Dense(16, activation='relu')`  
- `Dense(1, activation='sigmoid')`

Compiled with:
```python
optimizer='adam'
loss='binary_crossentropy'
metrics=['accuracy', 'precision', 'recall', 'AUC']
🏗️ Federated Learning Setup
Clients: 5

Rounds: 10

Local Epochs: 3

Batch Size: 64

FedAvg is used to aggregate client weights based on data contribution.

🔐 Differential Privacy
Laplacian noise is added to data samples for differential privacy:

python
Copy
Edit
def add_differential_privacy(X, epsilon=1.0):
    scale = 1.0 / epsilon
    noise = np.random.laplace(0, scale, X.shape)
    return X + noise
This ensures individual transaction data cannot be reverse-engineered.

📁 File Structure
Copy
Edit
federated-fraud-detection/
├── creditcard.csv
├── federated_learning.py
├── federated_fraud_detection_model.h5
├── federated_learning_metrics.png
├── confusion_matrix.png
├── roc_curve.png
└── README.md
🚀 How to Run
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/federated-fraud-detection.git
cd federated-fraud-detection
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is missing, generate it using:
pip freeze > requirements.txt

3. Run the Code
bash
Copy
Edit
python federated_learning.py
Make sure creditcard.csv is placed in the same directory.

🧪 Sample Output
csharp
Copy
Edit
Example of fraud detection on a new transaction:
Fraud probability: 0.0024
Transaction classified as: Legitimate
📊 Visual Outputs
federated_learning_metrics.png – Accuracy, Loss, Precision, Recall, AUC over rounds

confusion_matrix.png – Final confusion matrix

roc_curve.png – ROC curve with AUC

📚 References
Credit Card Fraud Dataset – Kaggle

Google AI Blog – Federated Learning

👨‍💻 Author
Aditya Sachin Gorave
B.Tech, Information Technology
📧 adityagorave2670@gmail.com
