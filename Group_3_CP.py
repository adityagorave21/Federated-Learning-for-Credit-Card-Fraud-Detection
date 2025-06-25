import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import random
import os
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

print("Loading dataset from local path...")
local_path = r"C:\Users\Aditya\Desktop\MLDL CP\creditcard.csv"
data = pd.read_csv(local_path)
print("Dataset loaded successfully!")

print(f"Dataset shape: {data.shape}")
print("\nSample of the dataset:")
print(data.head())
print("\nDataset info:")
print(data.info())
print("\nClass distribution:")
print(data['Class'].value_counts())

# Data preprocessing
def preprocess_data(df):
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

#  model architecture
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 
                          tf.keras.metrics.Precision(name='precision'),
                          tf.keras.metrics.Recall(name='recall'),
                          tf.keras.metrics.AUC(name='auc')])
    return model

# Simulate  clients 
def split_data_for_clients(X, y, num_clients, stratify=True):
    """Split data among multiple clients in a federated learning setting."""
    clients_data = []
    
    if stratify:
        
        fraud_indices = np.where(y == 1)[0]
        non_fraud_indices = np.where(y == 0)[0]
        
    
        fraud_per_client = len(fraud_indices) // num_clients
        non_fraud_per_client = len(non_fraud_indices) // num_clients
        
        for i in range(num_clients):
            if i < num_clients - 1:
                client_fraud_idx = fraud_indices[i*fraud_per_client:(i+1)*fraud_per_client]
                client_non_fraud_idx = non_fraud_indices[i*non_fraud_per_client:(i+1)*non_fraud_per_client]
            else:
                # Last client gets remaining data
                client_fraud_idx = fraud_indices[i*fraud_per_client:]
                client_non_fraud_idx = non_fraud_indices[i*non_fraud_per_client:]
                
            client_indices = np.concatenate([client_fraud_idx, client_non_fraud_idx])
            np.random.shuffle(client_indices)
            
            client_X = X[client_indices]
            client_y = y[client_indices]
            
            clients_data.append((client_X, client_y))
    else:
        # Simple random split 
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split_size = len(indices) // num_clients
        
        for i in range(num_clients):
            if i < num_clients - 1:
                client_indices = indices[i*split_size:(i+1)*split_size]
            else:
                client_indices = indices[i*split_size:]
                
            client_X = X[client_indices]
            client_y = y[client_indices]
            
            clients_data.append((client_X, client_y))
    
    return clients_data

# Convert labels to numpy arrays for tensorflow
y_train_np = y_train.values
y_test_np = y_test.values

# Define number of clients
num_clients = 5
print(f"Splitting data for {num_clients} clients...")
clients_data = split_data_for_clients(X_train, y_train_np, num_clients, stratify=True)

# Print information about client data distribution
for i, (client_X, client_y) in enumerate(clients_data):
    fraud_percent = np.sum(client_y) / len(client_y) * 100
    print(f"Client {i+1}: {len(client_X)} samples, {np.sum(client_y)} fraud cases ({fraud_percent:.2f}%)")

# Fixed Federated Learning implementation
def federated_learning(clients_data, global_model, num_rounds, local_epochs, batch_size, test_data):
    """
    Implementation of Federated Learning for credit card fraud detection.
    
    Parameters:
    - clients_data: List of tuples containing (X, y) for each client
    - global_model: Initial global model architecture
    - num_rounds: Number of federated learning rounds
    - local_epochs: Number of training epochs for each client in each round
    - batch_size: Batch size for training
    - test_data: Tuple of (X_test, y_test) for evaluation
    """
    X_test, y_test = test_data
    global_weights = global_model.get_weights()
    
    # History tracking for plotting
    history = {
        'global_accuracy': [],
        'global_loss': [],
        'global_precision': [],
        'global_recall': [],
        'global_auc': []
    }
    
    for round_num in range(num_rounds):
        print(f"\nFederated Learning Round {round_num+1}/{num_rounds}")
        
        # Store each client's weights and sample counts
        client_weights = []
        client_samples = []
        
        # Train on each client
        for client_id, (client_X, client_y) in enumerate(clients_data):
            print(f"Training on Client {client_id+1}/{len(clients_data)}")
            
            # Create and initialize client model with global weights
            client_model = create_model()
            client_model.set_weights(global_weights)
            
            # Train the client model
            client_model.fit(
                client_X, client_y,
                epochs=local_epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            # Store client weights and sample count for aggregation
            client_weights.append(client_model.get_weights())
            client_samples.append(len(client_X))
            
            # Evaluate client model on its own data
            client_eval = client_model.evaluate(client_X, client_y, verbose=0)
            print(f"  Client {client_id+1} - Loss: {client_eval[0]:.4f}, Accuracy: {client_eval[1]:.4f}, "
                  f"Precision: {client_eval[2]:.4f}, Recall: {client_eval[3]:.4f}, AUC: {client_eval[4]:.4f}")
        
        # Federated Averaging - Fixed implementation
        total_samples = sum(client_samples)
        
        # Initialize new global weights with the same structure as the model weights
        new_global_weights = [np.zeros_like(w) for w in global_weights]
        
        # Perform weighted averaging of client weights
        for client_idx, client_w in enumerate(client_weights):
            client_weight_ratio = client_samples[client_idx] / total_samples
            for layer_idx, layer_weights in enumerate(client_w):
                new_global_weights[layer_idx] += layer_weights * client_weight_ratio
        
        # Update global model
        global_model.set_weights(new_global_weights)
        
        # Evaluate global model
        global_metrics = global_model.evaluate(X_test, y_test, verbose=0)
        print(f"\nAfter round {round_num+1}:")
        print(f"Global model - Loss: {global_metrics[0]:.4f}, Accuracy: {global_metrics[1]:.4f}, "
              f"Precision: {global_metrics[2]:.4f}, Recall: {global_metrics[3]:.4f}, AUC: {global_metrics[4]:.4f}")
        
        # Save metrics for plotting
        history['global_loss'].append(global_metrics[0])
        history['global_accuracy'].append(global_metrics[1])
        history['global_precision'].append(global_metrics[2])
        history['global_recall'].append(global_metrics[3])
        history['global_auc'].append(global_metrics[4])
    
    return global_model, history

# Initialize the global model
global_model = create_model()

# Set FL parameters
num_rounds = 10
local_epochs = 3
batch_size = 64


print("\nStarting Federated Learning...")
final_global_model, history = federated_learning(
    clients_data, 
    global_model, 
    num_rounds, 
    local_epochs, 
    batch_size, 
    (X_test, y_test_np)
)


plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
plt.plot(range(1, num_rounds+1), history['global_accuracy'], marker='o', linestyle='-')
plt.title('Global Model Accuracy')
plt.xlabel('Federated Learning Round')
plt.ylabel('Accuracy')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(range(1, num_rounds+1), history['global_loss'], marker='o', linestyle='-', color='orange')
plt.title('Global Model Loss')
plt.xlabel('Federated Learning Round')
plt.ylabel('Loss')
plt.grid(True)

# Plot precision and recall
plt.subplot(2, 2, 3)
plt.plot(range(1, num_rounds+1), history['global_precision'], marker='o', linestyle='-', color='green')
plt.plot(range(1, num_rounds+1), history['global_recall'], marker='o', linestyle='-', color='red')
plt.title('Global Model Precision and Recall')
plt.xlabel('Federated Learning Round')
plt.ylabel('Score')
plt.legend(['Precision', 'Recall'])
plt.grid(True)

# Plot AUC
plt.subplot(2, 2, 4)
plt.plot(range(1, num_rounds+1), history['global_auc'], marker='o', linestyle='-', color='purple')
plt.title('Global Model AUC')
plt.xlabel('Federated Learning Round')
plt.ylabel('AUC')
plt.grid(True)

plt.tight_layout()
plt.savefig('federated_learning_metrics.png')
plt.show()

# Final evaluation of the global model
y_pred_proba = final_global_model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate metrics
print("\nFinal model evaluation:")
print(classification_report(y_test_np, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_np, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test_np, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()

# Save the model
final_global_model.save("federated_fraud_detection_model.h5")
print("\nModel saved as 'federated_fraud_detection_model.h5'")

# Simulating a privacy-preserving prediction
def private_prediction(new_transaction, model, scaler):
    """
    Simulate making a privacy-preserving prediction on a new transaction.
    In a real federated system, this would be done on the client side.
    """
    # Example transaction (replace with real data in production)
    if new_transaction is None:
        new_transaction = {
            'Time': 43590,
            'V1': -1.3598071336738,
            'V2': -0.0727811733098497,
            'V3': 2.53634673796914,
            'V4': 1.37815522427443,
            'V5': -0.338320769942518,
            'V6': 0.462387777762292,
            'V7': 0.239598554061257,
            'V8': 0.0986979012610507,
            'V9': 0.363786969611213,
            'V10': 0.0907941719789316,
            'V11': -0.551599533260813,
            'V12': -0.617800855762348,
            'V13': -0.991389847235408,
            'V14': -0.311169353699879,
            'V15': 1.46817697209427,
            'V16': -0.470400525259478,
            'V17': 0.207971241929242,
            'V18': 0.0257905801985591,
            'V19': 0.403992960255733,
            'V20': 0.251412098239705,
            'V21': -0.018306777944153,
            'V22': 0.277837575558899,
            'V23': -0.110473910188767,
            'V24': 0.0669280749146731,
            'V25': 0.128539358273528,
            'V26': -0.189114843888824,
            'V27': 0.133558376740387,
            'V28': -0.0210530534538215,
            'Amount': 149.62
        }
    
    # Convert to DataFrame and scale
    new_df = pd.DataFrame([new_transaction])
    if 'Class' in new_df.columns:
        new_df = new_df.drop('Class', axis=1)
    
    # Scale the features
    new_df_scaled = scaler.transform(new_df)
    
    # Make prediction
    prediction = model.predict(new_df_scaled, verbose=0)[0][0]
    
    return {
        'fraud_probability': float(prediction),
        'is_fraud': bool(prediction > 0.5)
    }

# Example of using the model for prediction
print("\nExample of fraud detection on a new transaction:")
prediction_result = private_prediction(None, final_global_model, scaler)
print(f"Fraud probability: {prediction_result['fraud_probability']:.4f}")
print(f"Transaction classified as: {'Fraudulent' if prediction_result['is_fraud'] else 'Legitimate'}")

# Function to simulate implementing differential privacy
def add_differential_privacy(X, epsilon=1.0):
    """
    Add Laplacian noise to implement differential privacy.
    
    Parameters:
    - X: Input data
    - epsilon: Privacy parameter (smaller = more privacy)
    
    Returns:
    - X with added noise
    """
    sensitivity = 1.0  # Assuming normalized data
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, X.shape)
    return X + noise

# Demonstrate differential privacy - this would be used in production systems
print("\nDemonstrating Differential Privacy:")
# Take a small sample for demonstration
sample_X = X_train[:5]
print("Original sample data:")
print(sample_X[:, :5])  # Show first 5 features only

# Add differential privacy
dp_X = add_differential_privacy(sample_X, epsilon=0.5)
print("\nSample data with differential privacy applied:")
print(dp_X[:, :5])  # Show first 5 features only

print("\nFederated Learning for Credit Card Fraud Detection project completed successfully!")