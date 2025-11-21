#!/usr/bin/env python3
"""
Inference script for EVO-based sequence classification

Uses a trained neural network classifier to predict classes for new sequences.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from joblib import load
import argparse
import os
import sys


def extract_embeddings(df, model_name, device, batch_size=1, pooling='mean'):
    """
    Extract embeddings from EVO model using monkey patching

    Args:
        df: DataFrame with 'sequence' column
        model_name: EVO model name
        device: Device to use (cuda or cpu)
        batch_size: Batch size for processing
        pooling: Pooling strategy ('mean' or 'max')

    Returns:
        embeddings: numpy array of embeddings
    """
    print(f"Loading EVO model: {model_name}...")

    try:
        from evo import Evo
    except ImportError:
        print("Error: evo-model package not found. Install with: pip install evo-model")
        sys.exit(1)

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

    print(f"Extracting embeddings for {len(df)} sequences...")
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size)):
            batch_df = df.iloc[i:i+batch_size]
            embeddings_batch = []

            for _, row in batch_df.iterrows():
                sequence = row['sequence']

                # Tokenize the sequence
                input_ids = torch.tensor(
                    tokenizer.tokenize(sequence),
                    dtype=torch.int
                ).unsqueeze(0).to(device)

                # Get embeddings (with monkey patch, the model output is now the embeddings)
                embed, _ = model(input_ids)

                # Apply pooling strategy
                if pooling == 'mean':
                    # Mean pooling
                    pooled_embedding = embed.mean(dim=1).to(torch.float32).cpu().numpy()
                elif pooling == 'max':
                    # Max pooling
                    pooled_embedding = torch.max(embed, dim=1)[0].to(torch.float32).cpu().numpy()
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling}")

                embeddings_batch.append(pooled_embedding)

            # Concatenate embeddings from the batch
            embeddings_batch = np.vstack(embeddings_batch)
            all_embeddings.append(embeddings_batch)

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    print(f"Embeddings extracted. Shape: {all_embeddings.shape}")

    return all_embeddings


def load_nn_classifier(model_path, device):
    """
    Load a trained neural network classifier

    Args:
        model_path: Path to saved model (.pt file)
        device: Device to use

    Returns:
        model: Loaded NN classifier
    """
    print(f"Loading trained model from {model_path}...")
    model = torch.load(model_path, map_location=device)
    model.device = torch.device(device)  # Update device
    print("  ✓ Model loaded successfully")
    return model


def predict(model, scaler, embeddings, device):
    """
    Make predictions using trained model

    Args:
        model: Trained NN classifier
        scaler: Fitted StandardScaler
        embeddings: Input embeddings
        device: Device to use

    Returns:
        predictions: Predicted class labels (0 or 1)
        probabilities: Predicted probabilities for class 1
    """
    print("Scaling embeddings...")
    embeddings_scaled = scaler.transform(embeddings)

    print("Making predictions...")
    predictions = model.predict(embeddings_scaled)
    probabilities = model.predict_proba(embeddings_scaled)[:, 1]

    return predictions, probabilities


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Predict sequence classes using trained EVO neural network classifier"
    )

    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV file with sequences")

    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV file for predictions")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained NN model (.pt file)")

    parser.add_argument("--scaler", type=str, required=True,
                        help="Path to fitted scaler (.joblib file)")

    parser.add_argument("--model_name", type=str, default="evo-1",
                        help="EVO model name to use (must match training)")

    parser.add_argument("--pooling", type=str, choices=["mean", "max"], default="mean",
                        help="Pooling strategy (must match training)")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for embedding extraction")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load input CSV
    print(f"Loading input file: {args.input}")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"  ✗ Error: Input file not found: {args.input}")
        sys.exit(1)

    print(f"  Found {len(df)} sequences")

    # Validate required columns
    required_cols = ['SeqID', 'start', 'end', 'sequence', 'label', 'Seq_Id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  ✗ Missing columns: {missing_cols}")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    # Extract embeddings
    embeddings = extract_embeddings(
        df,
        args.model_name,
        device,
        batch_size=args.batch_size,
        pooling=args.pooling
    )

    # Load scaler
    print(f"Loading scaler from {args.scaler}...")
    try:
        scaler = load(args.scaler)
    except FileNotFoundError:
        print(f"  ✗ Error: Scaler file not found: {args.scaler}")
        sys.exit(1)

    # Load model
    try:
        model = load_nn_classifier(args.model, device)
    except FileNotFoundError:
        print(f"  ✗ Error: Model file not found: {args.model}")
        sys.exit(1)

    # Make predictions
    predictions, probabilities = predict(model, scaler, embeddings, device)

    # Create output dataframe
    print("Preparing output...")
    df['predicted_label'] = predictions
    df['probability'] = probabilities  # P(phage) - probability of class 1
    df['confidence'] = np.maximum(probabilities, 1 - probabilities)  # P(predicted class)

    # Select output columns in desired order (matching ProkBERT format)
    output_df = df[['SeqID', 'start', 'end', 'label', 'predicted_label', 'probability', 'confidence', 'Seq_Id']]

    # Save predictions
    print(f"Saving predictions to {args.output}...")
    output_df.to_csv(args.output, index=False)

    # Calculate accuracy
    accuracy = (df['label'] == df['predicted_label']).mean()

    # Get basename of output file
    output_basename = os.path.basename(args.output)

    # Print summary (matching ProkBERT format)
    print(f"  ✓ Saved to: {output_basename}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Predicted labels: {df['predicted_label'].value_counts().to_dict()}")
    print(f"  Mean probability (phage): {np.mean(probabilities):.4f}")


if __name__ == "__main__":
    main()
