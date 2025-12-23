import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from joblib import dump, load
import os
import time
import argparse
import random
import hashlib
import glob

def compute_md5(sequence):
    """Compute MD5 hash of a DNA sequence"""
    return hashlib.md5(sequence.encode()).hexdigest()

def extract_embeddings(df, model_name, device, batch_size=1, pooling='mean', checkpoint_dir=None, checkpoint_interval=1024):
    """
    Extract embeddings from EVO model using monkey patching with checkpointing support

    Args:
        df: DataFrame with 'sequence', 'label', and 'md5' columns
        model_name: EVO model name
        device: Device to use
        batch_size: Batch size for processing
        pooling: Pooling strategy ('mean' or 'max')
        checkpoint_dir: Directory to save checkpoints (if None, no checkpointing)
        checkpoint_interval: Number of sequences between checkpoints

    Returns:
        embeddings, labels, md5s: Arrays of embeddings, labels, and MD5 hashes
    """
    # Check for existing checkpoints and resume if possible
    start_idx = 0
    all_embeddings = []
    all_labels = []
    all_md5s = []

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_pattern = os.path.join(checkpoint_dir, "chunk_*.npz")
        existing_checkpoints = sorted(glob.glob(checkpoint_pattern))

        if existing_checkpoints:
            print(f"Found {len(existing_checkpoints)} existing checkpoints. Loading...")
            for ckpt_path in existing_checkpoints:
                ckpt_data = np.load(ckpt_path)
                all_embeddings.append(ckpt_data['embeddings'])
                all_labels.extend(ckpt_data['labels'])
                all_md5s.extend(ckpt_data['md5s'])
                start_idx = len(all_labels)
            print(f"Resumed from checkpoint. Starting at index {start_idx}/{len(df)}")

            if start_idx >= len(df):
                print("All embeddings already extracted!")
                all_embeddings = np.vstack(all_embeddings)
                return all_embeddings, np.array(all_labels), np.array(all_md5s)

    print(f"Loading EVO model: {model_name}...")
    try:
        from evo import Evo
    except ImportError:
        print("Error: evo-model package not found. Install with: pip install evo-model")
        exit(1)

    evo_model = Evo(model_name)
    model, tokenizer = evo_model.model, evo_model.tokenizer
    model.to(device)
    model.eval()

    # Apply the monkey patch to get embeddings instead of logits
    print("Applying monkey patch to obtain embeddings...")

    class CustomEmbedding(nn.Module):
        def unembed(self, u):
            return u

    model.unembed = CustomEmbedding()

    print(f"Extracting embeddings from index {start_idx} to {len(df)}...")
    current_chunk_embeddings = []
    current_chunk_labels = []
    current_chunk_md5s = []
    checkpoint_counter = start_idx // checkpoint_interval

    with torch.no_grad():
        for i in tqdm(range(start_idx, len(df), batch_size)):
            batch_df = df.iloc[i:i+batch_size]
            embeddings_batch = []
            labels_batch = []
            md5s_batch = []

            for _, row in batch_df.iterrows():
                sequence = row['sequence']
                label = row['label']
                md5_hash = row['md5']

                # Tokenize the sequence
                input_ids = torch.tensor(
                    tokenizer.tokenize(sequence),
                    dtype=torch.int
                ).unsqueeze(0).to(device)

                # Get embeddings (with monkey patch, the model output is now the embeddings)
                embed, _ = model(input_ids)

                # Apply pooling strategy
                if pooling == 'mean':
                    pooled_embedding = embed.mean(dim=1).to(torch.float32).cpu().numpy()
                elif pooling == 'max':
                    pooled_embedding = torch.max(embed, dim=1)[0].to(torch.float32).cpu().numpy()
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling}")

                embeddings_batch.append(pooled_embedding)
                labels_batch.append(label)
                md5s_batch.append(md5_hash)

            # Add to current chunk
            embeddings_batch = np.vstack(embeddings_batch)
            current_chunk_embeddings.append(embeddings_batch)
            current_chunk_labels.extend(labels_batch)
            current_chunk_md5s.extend(md5s_batch)

            # Save checkpoint if interval reached
            if checkpoint_dir is not None and len(current_chunk_labels) >= checkpoint_interval:
                chunk_embeddings = np.vstack(current_chunk_embeddings)
                checkpoint_path = os.path.join(checkpoint_dir, f"chunk_{checkpoint_counter:04d}.npz")
                np.savez(checkpoint_path,
                        embeddings=chunk_embeddings,
                        labels=np.array(current_chunk_labels),
                        md5s=np.array(current_chunk_md5s))
                print(f"\nSaved checkpoint: {checkpoint_path} ({len(current_chunk_labels)} sequences)")

                # Add to all_embeddings and reset current chunk
                all_embeddings.append(chunk_embeddings)
                all_labels.extend(current_chunk_labels)
                all_md5s.extend(current_chunk_md5s)

                current_chunk_embeddings = []
                current_chunk_labels = []
                current_chunk_md5s = []
                checkpoint_counter += 1

    # Save remaining sequences as final checkpoint
    if current_chunk_embeddings:
        chunk_embeddings = np.vstack(current_chunk_embeddings)
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, f"chunk_{checkpoint_counter:04d}.npz")
            np.savez(checkpoint_path,
                    embeddings=chunk_embeddings,
                    labels=np.array(current_chunk_labels),
                    md5s=np.array(current_chunk_md5s))
            print(f"\nSaved final checkpoint: {checkpoint_path} ({len(current_chunk_labels)} sequences)")

        all_embeddings.append(chunk_embeddings)
        all_labels.extend(current_chunk_labels)
        all_md5s.extend(current_chunk_md5s)

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)
    all_md5s = np.array(all_md5s)
    print(f"Embeddings extracted. Shape: {all_embeddings.shape}")

    return all_embeddings, all_labels, all_md5s
def train_linear_classifier(X_train, y_train, X_val=None, y_val=None, C_values=None):
    """
    Train a logistic regression classifier on the embeddings

    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_val: Validation embeddings (optional)
        y_val: Validation labels (optional)
        C_values: List of C values to try (optional)

    Returns:
        best_model: Trained logistic regression model
    """
    print("Training linear classifier (Logistic Regression)...")
    print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1, 10, 100]

    # If validation set is provided, use grid search to find best hyperparameters
    if X_val is not None and y_val is not None:
        param_grid = {'C': C_values}
        grid_search = GridSearchCV(
            LogisticRegression(max_iter=1000, class_weight='balanced', verbose=0),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=2  # Show progress
        )
        print(f"Starting GridSearchCV with {len(C_values)} C values and 5-fold CV...")
        print(f"Total fits: {len(C_values) * 5}")
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"\nBest parameters: {grid_search.best_params_}")

        # Evaluate on validation set
        y_val_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_mcc = matthews_corrcoef(y_val, y_val_pred)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 score: {val_f1:.4f}")
        print(f"Validation MCC: {val_mcc:.4f}")
    else:
        # Train with default parameters
        print("Training with default C=1.0...")
        best_model = LogisticRegression(max_iter=1000, class_weight='balanced', verbose=1)
        best_model.fit(X_train, y_train)

    return best_model

def train_svm_classifier(X_train, y_train, X_val=None, y_val=None, C_values=None, simple_mode=False):
    """
    Train an SVM classifier on the embeddings

    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_val: Validation embeddings (optional)
        y_val: Validation labels (optional)
        C_values: List of C values to try (optional)
        simple_mode: If True, skip GridSearchCV and use fixed C=1.0 (saves memory)

    Returns:
        best_model: Trained SVM model
    """
    print("Training SVM classifier with linear kernel...")
    print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Simple mode: skip hyperparameter search to save memory
    if simple_mode:
        print("Running in simple mode (no hyperparameter search to save memory)...")
        print("Using fixed C=1.0")
        best_model = SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced',
                        cache_size=2000, verbose=True)
        best_model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            # Evaluate on validation set
            y_val_pred = best_model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred)
            val_mcc = matthews_corrcoef(y_val, y_val_pred)
            print(f"Validation accuracy: {val_accuracy:.4f}")
            print(f"Validation F1 score: {val_f1:.4f}")
            print(f"Validation MCC: {val_mcc:.4f}")

        return best_model

    # Regular mode: use GridSearchCV for hyperparameter search
    if C_values is None:
        C_values = [0.01, 0.1, 1, 10]  # Reduced from 6 to 4 values to save memory

    # If validation set is provided, use grid search to find best hyperparameters
    if X_val is not None and y_val is not None:
        param_grid = {
            'C': C_values,
            'kernel': ['linear']
        }
        grid_search = GridSearchCV(
            SVC(probability=True, class_weight='balanced', cache_size=1000),  # Increased cache
            param_grid,
            cv=3,  # Reduced from 5 to 3 folds to save memory
            scoring='f1',
            n_jobs=4,  # Reduced from -1 to limit parallel workers
            verbose=2  # Show progress
        )
        print(f"Starting GridSearchCV with {len(C_values)} C values and 3-fold CV...")
        print(f"Total fits: {len(C_values) * 3}")
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")

        # Evaluate on validation set
        y_val_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_mcc = matthews_corrcoef(y_val, y_val_pred)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 score: {val_f1:.4f}")
        print(f"Validation MCC: {val_mcc:.4f}")
    else:
        # Train with default parameters
        best_model = SVC(kernel='linear', probability=True, class_weight='balanced', cache_size=1000)
        best_model.fit(X_train, y_train)

    return best_model

class ThreeLayerNN(nn.Module):
    """
    Three-layer neural network for binary classification

    Architecture:
    - Input layer: input_dim
    - Hidden layer 1: hidden_dim1 with ReLU activation and Dropout
    - Hidden layer 2: hidden_dim2 with ReLU activation and Dropout
    - Output layer: 1 (sigmoid activation for binary classification)
    """
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, dropout=0.3):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class NeuralNetworkClassifier:
    """
    Wrapper class for PyTorch neural network to provide scikit-learn-like interface
    """
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, dropout=0.3,
                 learning_rate=0.001, epochs=100, batch_size=32, device='cpu',
                 early_stopping_patience=10, use_wandb=False):
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.early_stopping_patience = early_stopping_patience
        self.use_wandb = use_wandb
        self.model = None
        self.criterion = nn.BCELoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}

    def fit(self, X, y, X_val=None, y_val=None):
        """Train the neural network with validation metrics and early stopping"""
        # Initialize model
        self.model = ThreeLayerNN(
            self.input_dim,
            self.hidden_dim1,
            self.hidden_dim2,
            self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        # Convert validation data if provided
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            perm = torch.randperm(X_tensor.size(0))
            X_shuffled = X_tensor[perm]
            y_shuffled = y_tensor[perm]

            epoch_loss = 0
            num_batches = 0
            for i in range(0, len(X_shuffled), self.batch_size):
                batch_X = X_shuffled[i:i+self.batch_size]
                batch_y = y_shuffled[i:i+self.batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches
            self.history['train_loss'].append(avg_train_loss)

            # Validation phase
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                    val_preds = (val_outputs.cpu().numpy() > 0.5).astype(int).flatten()
                    val_accuracy = accuracy_score(y_val, val_preds)
                    val_f1 = f1_score(y_val, val_preds)

                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                self.history['val_f1'].append(val_f1)

                # Print progress
                print(f"Epoch [{epoch+1}/{self.epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_accuracy:.4f}, "
                      f"Val F1: {val_f1:.4f}")

                # Log to wandb if enabled
                if self.use_wandb:
                    try:
                        import wandb
                        wandb.log({
                            'epoch': epoch + 1,
                            'train_loss': avg_train_loss,
                            'val_loss': val_loss,
                            'val_accuracy': val_accuracy,
                            'val_f1': val_f1
                        })
                    except ImportError:
                        pass

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        # Restore best model
                        if best_model_state is not None:
                            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
                        break
            else:
                # No validation data, just print training loss
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}")

        return self

    def predict(self, X):
        """Predict class labels"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
        return predictions

    def predict_proba(self, X):
        """Predict class probabilities"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            proba = outputs.cpu().numpy().flatten()
        # Return probabilities for both classes
        return np.column_stack([1 - proba, proba])

def train_nn_classifier(X_train, y_train, X_val=None, y_val=None, device='cpu', use_wandb=False):
    """
    Train a 3-layer neural network classifier on the embeddings

    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_val: Validation embeddings (optional)
        y_val: Validation labels (optional)
        device: Device to use for training ('cpu' or 'cuda')
        use_wandb: Whether to log to wandb (default: False)

    Returns:
        best_model: Trained neural network model
    """
    print("Training 3-layer neural network classifier...")
    print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Device: {device}")

    input_dim = X_train.shape[1]

    # Hyperparameter grid for manual search
    param_combinations = [
        {'hidden_dim1': 256, 'hidden_dim2': 128, 'dropout': 0.3, 'learning_rate': 0.001, 'epochs': 200},
        {'hidden_dim1': 512, 'hidden_dim2': 256, 'dropout': 0.3, 'learning_rate': 0.001, 'epochs': 200},
        {'hidden_dim1': 256, 'hidden_dim2': 128, 'dropout': 0.5, 'learning_rate': 0.0001, 'epochs': 200},
    ]

    best_model = None
    best_f1 = 0

    if X_val is not None and y_val is not None:
        print(f"Performing hyperparameter search with {len(param_combinations)} configurations...")
        print(f"Early stopping enabled with patience=10")

        for i, params in enumerate(param_combinations):
            print(f"\n{'='*60}")
            print(f"Configuration {i+1}/{len(param_combinations)}:")
            print(f"  Architecture: {params['hidden_dim1']} -> {params['hidden_dim2']} -> 1")
            print(f"  Dropout: {params['dropout']}")
            print(f"  Learning rate: {params['learning_rate']}")
            print(f"  Max Epochs: {params['epochs']}")
            print(f"{'='*60}")

            # Initialize wandb run if enabled
            if use_wandb:
                try:
                    import wandb
                    wandb.init(
                        project="evo-classifier",
                        name=f"nn_config_{i+1}",
                        config=params,
                        reinit=True
                    )
                except ImportError:
                    print("Warning: wandb not installed. Skipping wandb logging.")
                    use_wandb = False

            model = NeuralNetworkClassifier(
                input_dim=input_dim,
                hidden_dim1=params['hidden_dim1'],
                hidden_dim2=params['hidden_dim2'],
                dropout=params['dropout'],
                learning_rate=params['learning_rate'],
                epochs=params['epochs'],
                batch_size=32,
                device=device,
                early_stopping_patience=10,
                use_wandb=use_wandb
            )

            model.fit(X_train, y_train, X_val, y_val)

            # Evaluate on validation set
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred)
            val_mcc = matthews_corrcoef(y_val, y_val_pred)

            print(f"\nFinal Validation Results:")
            print(f"  Accuracy: {val_accuracy:.4f}")
            print(f"  F1 score: {val_f1:.4f}")
            print(f"  MCC: {val_mcc:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model = model
                print(f"  *** New best model! F1: {best_f1:.4f} ***")

            # Finish wandb run
            if use_wandb:
                try:
                    import wandb
                    wandb.finish()
                except:
                    pass

        print(f"\n{'='*60}")
        print(f"Best validation F1 score: {best_f1:.4f}")
        print(f"{'='*60}")
    else:
        # Train with default parameters
        print("Training with default hyperparameters...")
        print(f"  Architecture: 256 -> 128 -> 1")
        print(f"  Dropout: 0.3")
        print(f"  Learning rate: 0.001")
        print(f"  Max Epochs: 200 (with early stopping)")

        if use_wandb:
            try:
                import wandb
                wandb.init(project="evo-classifier", name="nn_default")
            except ImportError:
                print("Warning: wandb not installed. Skipping wandb logging.")
                use_wandb = False

        best_model = NeuralNetworkClassifier(
            input_dim=input_dim,
            hidden_dim1=256,
            hidden_dim2=128,
            dropout=0.3,
            learning_rate=0.001,
            epochs=200,
            batch_size=32,
            device=device,
            early_stopping_patience=10,
            use_wandb=use_wandb
        )
        best_model.fit(X_train, y_train, X_val, y_val)

        if use_wandb:
            try:
                import wandb
                wandb.finish()
            except:
                pass

    return best_model

def evaluate_model(model, X_test, y_test, md5s_test=None, model_name="Model", output_dir="."):
    """
    Evaluate the model on the test set and save per-sample predictions

    Args:
        model: Trained classifier
        X_test: Test embeddings
        y_test: Test labels
        md5s_test: Test MD5 hashes (optional)
        model_name: Name of the model for printing
        output_dir: Directory to save predictions CSV

    Returns:
        results: Dictionary with evaluation metrics
    """
    print(f"Evaluating {model_name}...")

    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)  # Calculate MCC
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")  # Print MCC
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Save per-sample predictions
    if md5s_test is not None:
        predictions_df = pd.DataFrame({
            'md5': md5s_test,
            'true_label': y_test,
            'predicted_label': y_pred,
            'probability': y_pred_proba
        })

        os.makedirs(output_dir, exist_ok=True)
        predictions_path = os.path.join(output_dir, f'predictions_{model_name.replace(" ", "_")}.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Saved predictions to {predictions_path}")

    # Store results in dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,  # Store MCC
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

    return results

def visualize_confusion_matrix(conf_matrix, model_name="Model", output_dir="."):
    """
    Visualize the confusion matrix
    
    Args:
        conf_matrix: Confusion matrix
        model_name: Name of the model for the title
        output_dir: Directory to save the confusion matrix image
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bacteria', 'Phage'],
                yticklabels=['Bacteria', 'Phage'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png')

    plt.savefig(output_path)
    print(f"Saved confusion matrix to {output_path}")
    plt.close()

def measure_silhouette(embeddings, labels, output_dir="."):
    """
    Measure the silhouette score of the embeddings in both original and PCA space

    Args:
        embeddings: Embeddings matrix (high-dimensional)
        labels: Labels array
        output_dir: Directory to save PCA visualization

    Returns:
        silhouette_original: Silhouette score in original space
        silhouette_pca: Silhouette score in PCA space (2 components)
    """
    print("Calculating silhouette scores...")

    # Original high-dimensional silhouette
    silhouette_original = silhouette_score(embeddings, labels)
    print(f"Silhouette Score (Original {embeddings.shape[1]}D): {silhouette_original:.4f}")

    # PCA to 2 components
    print("Performing PCA (2 components) for visualization and silhouette...")
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)

    # PCA silhouette
    silhouette_pca = silhouette_score(embeddings_pca, labels)
    print(f"Silhouette Score (PCA 2D): {silhouette_pca:.4f}")
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_[0]:.4f} + {pca.explained_variance_ratio_[1]:.4f})")

    # Create PCA visualization
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e']  # Blue for bacteria, orange for phage
    labels_names = ['Bacteria', 'Phage']

    for label_val in np.unique(labels):
        mask = labels == label_val
        plt.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1],
                   c=colors[label_val], label=labels_names[label_val],
                   alpha=0.6, s=20, edgecolors='k', linewidth=0.5)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.title(f'PCA Visualization of EVO Embeddings\nSilhouette Score: {silhouette_pca:.4f}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save PCA plot
    os.makedirs(output_dir, exist_ok=True)
    pca_plot_path = os.path.join(output_dir, 'pca_visualization.png')
    plt.savefig(pca_plot_path, dpi=300)
    print(f"PCA visualization saved to {pca_plot_path}")
    plt.close()

    return silhouette_original, silhouette_pca

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="EVO Embeddings Classifier for Phage-Bacteria Classification")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing train.csv, dev.csv, and test.csv files")

    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results, models, and visualizations")

    parser.add_argument("--pooling", type=str, choices=["mean", "max"], default="mean",
                        help="Pooling strategy for embeddings (mean or max)")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for embedding extraction")

    parser.add_argument("--model_name", type=str, default="evo-1",
                        help="EVO model name to use")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")

    parser.add_argument("--classifiers", type=str, nargs='+',
                        choices=['linear', 'svm', 'nn', 'all'], default=['all'],
                        help="Classifiers to train: linear, svm, nn, or all (default: all)")

    parser.add_argument("--embeddings_dir", type=str, default=None,
                        help="Directory to load/save embeddings (default: same as output_dir)")

    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: None, uses random seed)")

    parser.add_argument("--svm_simple_mode", action='store_true',
                        help="Run SVM in simple mode (no hyperparameter search, saves memory)")

    parser.add_argument("--silhouette_only", action='store_true',
                        help="Only calculate silhouette scores and create PCA visualization (skip training)")

    parser.add_argument("--use_wandb", action='store_true',
                        help="Enable wandb logging for training (requires wandb to be installed)")

    parser.add_argument("--checkpoint_interval", type=int, default=1024,
                        help="Number of sequences between embedding extraction checkpoints (default: 1024)")

    return parser.parse_args()

def set_random_seed(seed):
    """
    Set random seed for reproducibility across all libraries

    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may impact performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Set random seed if provided
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        set_random_seed(args.seed)
    else:
        print("No seed specified - results will not be reproducible")

    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load datasets
    print("Loading datasets...")
    try:
        train_path = os.path.join(args.input_dir, "train.csv")
        dev_path = os.path.join(args.input_dir, "dev.csv")
        test_path = os.path.join(args.input_dir, "test.csv")

        train_df = pd.read_csv(train_path)
        dev_df = pd.read_csv(dev_path)
        test_df = pd.read_csv(test_path)

        print(f"Train set loaded. Shape: {train_df.shape}")
        print(f"Dev set loaded. Shape: {dev_df.shape}")
        print(f"Test set loaded. Shape: {test_df.shape}")

        # Verify that the dataframes have the expected columns
        required_columns = ['sequence', 'label']
        for df_name, df in [("Train", train_df), ("Dev", dev_df), ("Test", test_df)]:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Error: {df_name} dataset is missing required columns: {missing_columns}")
                exit(1)

        # Compute MD5 hashes if not already present
        print("\nComputing MD5 hashes for sequences...")
        for df_name, df in [("Train", train_df), ("Dev", dev_df), ("Test", test_df)]:
            if 'md5' not in df.columns:
                print(f"  Computing MD5s for {df_name} set...")
                df['md5'] = df['sequence'].apply(compute_md5)
            else:
                print(f"  {df_name} set already has MD5 column")

            # Check for duplicates
            n_total = len(df)
            n_unique = df['md5'].nunique()
            n_duplicates = n_total - n_unique
            if n_duplicates > 0:
                print(f"  WARNING: {df_name} set contains {n_duplicates} duplicate sequences!")
                print(f"    Total: {n_total}, Unique: {n_unique}")
            else:
                print(f"  {df_name} set: All {n_total} sequences are unique")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure train.csv, dev.csv, and test.csv exist in the specified input directory.")
        exit(1)

    # Determine embeddings directory (use separate dir if specified, otherwise use output_dir)
    embeddings_dir = args.embeddings_dir if args.embeddings_dir is not None else args.output_dir
    os.makedirs(embeddings_dir, exist_ok=True)

    # Create paths for saved embeddings and checkpoint directories
    train_embed_path = os.path.join(embeddings_dir, f"train_embeddings_{args.pooling}.npz")
    dev_embed_path = os.path.join(embeddings_dir, f"dev_embeddings_{args.pooling}.npz")
    test_embed_path = os.path.join(embeddings_dir, f"test_embeddings_{args.pooling}.npz")

    train_checkpoint_dir = os.path.join(embeddings_dir, f"train_checkpoints_{args.pooling}")
    dev_checkpoint_dir = os.path.join(embeddings_dir, f"dev_checkpoints_{args.pooling}")
    test_checkpoint_dir = os.path.join(embeddings_dir, f"test_checkpoints_{args.pooling}")

    # Extract or load embeddings for train set
    print("\n" + "="*60)
    print("TRAIN SET EMBEDDINGS")
    print("="*60)
    if os.path.exists(train_embed_path):
        print(f"Loading pre-extracted train embeddings from {train_embed_path}...")
        train_data = np.load(train_embed_path, allow_pickle=True)
        X_train, y_train, md5s_train = train_data['embeddings'], train_data['labels'], train_data['md5s']
    else:
        print("Extracting embeddings for train set...")
        X_train, y_train, md5s_train = extract_embeddings(
            train_df, args.model_name, device,
            batch_size=args.batch_size,
            pooling=args.pooling,
            checkpoint_dir=train_checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval
        )
        # Save final embeddings
        np.savez(train_embed_path, embeddings=X_train, labels=y_train, md5s=md5s_train)
        print(f"Train embeddings saved to {train_embed_path}")

    # Extract or load embeddings for dev set
    print("\n" + "="*60)
    print("DEV SET EMBEDDINGS")
    print("="*60)
    if os.path.exists(dev_embed_path):
        print(f"Loading pre-extracted dev embeddings from {dev_embed_path}...")
        dev_data = np.load(dev_embed_path, allow_pickle=True)
        X_val, y_val, md5s_val = dev_data['embeddings'], dev_data['labels'], dev_data['md5s']
    else:
        print("Extracting embeddings for dev set...")
        X_val, y_val, md5s_val = extract_embeddings(
            dev_df, args.model_name, device,
            batch_size=args.batch_size,
            pooling=args.pooling,
            checkpoint_dir=dev_checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval
        )
        # Save final embeddings
        np.savez(dev_embed_path, embeddings=X_val, labels=y_val, md5s=md5s_val)
        print(f"Dev embeddings saved to {dev_embed_path}")

    # Extract or load embeddings for test set
    print("\n" + "="*60)
    print("TEST SET EMBEDDINGS")
    print("="*60)
    if os.path.exists(test_embed_path):
        print(f"Loading pre-extracted test embeddings from {test_embed_path}...")
        test_data = np.load(test_embed_path, allow_pickle=True)
        X_test, y_test, md5s_test = test_data['embeddings'], test_data['labels'], test_data['md5s']
    else:
        print("Extracting embeddings for test set...")
        X_test, y_test, md5s_test = extract_embeddings(
            test_df, args.model_name, device,
            batch_size=args.batch_size,
            pooling=args.pooling,
            checkpoint_dir=test_checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval
        )
        # Save final embeddings
        np.savez(test_embed_path, embeddings=X_test, labels=y_test, md5s=md5s_test)
        print(f"Test embeddings saved to {test_embed_path}")

    print(f"Train embeddings shape: {X_train.shape}")
    print(f"Dev embeddings shape: {X_val.shape}")
    print(f"Test embeddings shape: {X_test.shape}")

    # Calculate silhouette score on training data (both original and PCA)
    silhouette_original, silhouette_pca = measure_silhouette(X_train, y_train, args.output_dir)

    # If silhouette_only mode, save results and exit
    if args.silhouette_only:
        print("\n" + "="*50)
        print("Silhouette-only mode: Skipping classifier training")
        print("="*50)

        # Save silhouette scores to a file
        silhouette_summary_path = os.path.join(args.output_dir, f'silhouette_scores_{args.pooling}.txt')
        with open(silhouette_summary_path, 'w') as f:
            f.write(f"EVO Embeddings Silhouette Analysis (Pooling: {args.pooling})\n")
            f.write("="*50 + "\n\n")
            f.write(f"Training set size: {X_train.shape[0]} samples\n")
            f.write(f"Embedding dimensions: {X_train.shape[1]}D\n\n")
            f.write(f"Silhouette Score (Original {X_train.shape[1]}D): {silhouette_original:.4f}\n")
            f.write(f"Silhouette Score (PCA 2D): {silhouette_pca:.4f}\n\n")
            f.write("Note: PCA visualization saved to pca_visualization.png\n")

        print(f"\nSilhouette scores saved to {silhouette_summary_path}")
        print("PCA visualization saved to pca_visualization.png")
        print("\nDone! Exiting without training classifiers.")
        return

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Determine which classifiers to train
    train_all = 'all' in args.classifiers
    train_linear = 'linear' in args.classifiers or train_all
    train_svm = 'svm' in args.classifiers or train_all
    train_nn = 'nn' in args.classifiers or train_all

    # Train Linear Classifier (Logistic Regression)
    if train_linear:
        start_time = time.time()
        linear_model = train_linear_classifier(X_train_scaled, y_train, X_val_scaled, y_val)
        linear_training_time = time.time() - start_time
        print(f"Linear classifier training time: {linear_training_time:.2f} seconds")
    else:
        linear_model = None
        linear_training_time = 0
        print("Skipping Linear classifier...")

    # Train SVM Classifier
    if train_svm:
        start_time = time.time()
        svm_model = train_svm_classifier(X_train_scaled, y_train, X_val_scaled, y_val,
                                        simple_mode=args.svm_simple_mode)
        svm_training_time = time.time() - start_time
        print(f"SVM training time: {svm_training_time:.2f} seconds")
    else:
        svm_model = None
        svm_training_time = 0
        print("Skipping SVM classifier...")

    # Train Neural Network Classifier
    if train_nn:
        start_time = time.time()
        nn_model = train_nn_classifier(X_train_scaled, y_train, X_val_scaled, y_val,
                                       device=args.device, use_wandb=args.use_wandb)
        nn_training_time = time.time() - start_time
        print(f"Neural Network training time: {nn_training_time:.2f} seconds")
    else:
        nn_model = None
        nn_training_time = 0
        print("Skipping Neural Network classifier...")

    # Evaluate models on test set
    linear_results = evaluate_model(linear_model, X_test_scaled, y_test,
                                    md5s_test=md5s_test, model_name="Logistic Regression",
                                    output_dir=args.output_dir) if train_linear else None
    svm_results = evaluate_model(svm_model, X_test_scaled, y_test,
                                 md5s_test=md5s_test, model_name="SVM (Linear Kernel)",
                                 output_dir=args.output_dir) if train_svm else None
    nn_results = evaluate_model(nn_model, X_test_scaled, y_test,
                                md5s_test=md5s_test, model_name="3-Layer Neural Network",
                                output_dir=args.output_dir) if train_nn else None

    # Visualize confusion matrices
    if train_linear:
        visualize_confusion_matrix(linear_results['confusion_matrix'], "Logistic Regression", args.output_dir)
    if train_svm:
        visualize_confusion_matrix(svm_results['confusion_matrix'], "SVM (Linear Kernel)", args.output_dir)
    if train_nn:
        visualize_confusion_matrix(nn_results['confusion_matrix'], "3-Layer Neural Network", args.output_dir)

    # Save models
    print("Saving models...")
    if train_linear:
        dump(linear_model, os.path.join(args.output_dir, 'evo_linear_classifier.joblib'))
    if train_svm:
        dump(svm_model, os.path.join(args.output_dir, 'evo_svm_classifier.joblib'))
    if train_nn:
        torch.save(nn_model, os.path.join(args.output_dir, 'evo_nn_classifier.pt'))
    dump(scaler, os.path.join(args.output_dir, 'evo_scaler.joblib'))

    # Determine the better model
    models_f1 = {}
    models_results = {}

    if train_linear:
        models_f1["Logistic Regression"] = linear_results['f1']
        models_results["Logistic Regression"] = linear_results
    if train_svm:
        models_f1["SVM (Linear Kernel)"] = svm_results['f1']
        models_results["SVM (Linear Kernel)"] = svm_results
    if train_nn:
        models_f1["3-Layer Neural Network"] = nn_results['f1']
        models_results["3-Layer Neural Network"] = nn_results

    if models_f1:
        better_model = max(models_f1, key=models_f1.get)
        better_results = models_results[better_model]

        print("\n" + "="*50)
        print(f"Best model: {better_model}")
        print(f"F1 Score: {better_results['f1']:.4f}")
        print(f"MCC: {better_results['mcc']:.4f}")
        print(f"Accuracy: {better_results['accuracy']:.4f}")
        print("="*50)

    # Export results to CSV
    if models_results:
        results_data = {
            'Model': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1 Score': [],
            'MCC': [],
            'ROC AUC': [],
            'Training Time (s)': []
        }

        if train_linear:
            results_data['Model'].append('Logistic Regression')
            results_data['Accuracy'].append(linear_results['accuracy'])
            results_data['Precision'].append(linear_results['precision'])
            results_data['Recall'].append(linear_results['recall'])
            results_data['F1 Score'].append(linear_results['f1'])
            results_data['MCC'].append(linear_results['mcc'])
            results_data['ROC AUC'].append(linear_results['roc_auc'])
            results_data['Training Time (s)'].append(linear_training_time)

        if train_svm:
            results_data['Model'].append('SVM (Linear Kernel)')
            results_data['Accuracy'].append(svm_results['accuracy'])
            results_data['Precision'].append(svm_results['precision'])
            results_data['Recall'].append(svm_results['recall'])
            results_data['F1 Score'].append(svm_results['f1'])
            results_data['MCC'].append(svm_results['mcc'])
            results_data['ROC AUC'].append(svm_results['roc_auc'])
            results_data['Training Time (s)'].append(svm_training_time)

        if train_nn:
            results_data['Model'].append('3-Layer Neural Network')
            results_data['Accuracy'].append(nn_results['accuracy'])
            results_data['Precision'].append(nn_results['precision'])
            results_data['Recall'].append(nn_results['recall'])
            results_data['F1 Score'].append(nn_results['f1'])
            results_data['MCC'].append(nn_results['mcc'])
            results_data['ROC AUC'].append(nn_results['roc_auc'])
            results_data['Training Time (s)'].append(nn_training_time)

        results_df = pd.DataFrame(results_data)
        results_csv_path = os.path.join(args.output_dir, f'evo_classifier_results_{args.pooling}.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to {results_csv_path}")

    # Save a summary text file
    summary_path = os.path.join(args.output_dir, f'summary_{args.pooling}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"EVO Classifier Summary (Pooling: {args.pooling})\n")
        f.write("="*50 + "\n\n")
        f.write(f"Silhouette Score (Original {X_train.shape[1]}D): {silhouette_original:.4f}\n")
        f.write(f"Silhouette Score (PCA 2D): {silhouette_pca:.4f}\n\n")

        if train_linear:
            f.write("Logistic Regression Results:\n")
            f.write(f"Accuracy: {linear_results['accuracy']:.4f}\n")
            f.write(f"Precision: {linear_results['precision']:.4f}\n")
            f.write(f"Recall: {linear_results['recall']:.4f}\n")
            f.write(f"F1 Score: {linear_results['f1']:.4f}\n")
            f.write(f"MCC: {linear_results['mcc']:.4f}\n")
            f.write(f"ROC AUC: {linear_results['roc_auc']:.4f}\n")
            f.write(f"Training Time: {linear_training_time:.2f} seconds\n\n")

        if train_svm:
            f.write("SVM Results:\n")
            f.write(f"Accuracy: {svm_results['accuracy']:.4f}\n")
            f.write(f"Precision: {svm_results['precision']:.4f}\n")
            f.write(f"Recall: {svm_results['recall']:.4f}\n")
            f.write(f"F1 Score: {svm_results['f1']:.4f}\n")
            f.write(f"MCC: {svm_results['mcc']:.4f}\n")
            f.write(f"ROC AUC: {svm_results['roc_auc']:.4f}\n")
            f.write(f"Training Time: {svm_training_time:.2f} seconds\n\n")

        if train_nn:
            f.write("3-Layer Neural Network Results:\n")
            f.write(f"Accuracy: {nn_results['accuracy']:.4f}\n")
            f.write(f"Precision: {nn_results['precision']:.4f}\n")
            f.write(f"Recall: {nn_results['recall']:.4f}\n")
            f.write(f"F1 Score: {nn_results['f1']:.4f}\n")
            f.write(f"MCC: {nn_results['mcc']:.4f}\n")
            f.write(f"ROC AUC: {nn_results['roc_auc']:.4f}\n")
            f.write(f"Training Time: {nn_training_time:.2f} seconds\n\n")

        if models_f1:
            f.write("="*50 + "\n")
            f.write(f"Best model: {better_model}\n")
            f.write(f"Best F1 Score: {better_results['f1']:.4f}\n")
            f.write(f"Best MCC: {better_results['mcc']:.4f}\n")
            f.write(f"Best Accuracy: {better_results['accuracy']:.4f}\n")

    print(f"Summary saved to {summary_path}")

    # Output summary for paper
    if models_f1:
        print("\nSummary for Paper:")
        print(f"EVO embeddings with {args.pooling} pooling demonstrated class separation (silhouette score: {silhouette_pca:.4f} in PCA space).")
        print(f"The best performing classifier was {better_model} with an accuracy of {better_results['accuracy']:.4f}, F1 score of {better_results['f1']:.4f}, and MCC of {better_results['mcc']:.4f}.")

if __name__ == "__main__":
    main()

