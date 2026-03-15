import os
import argparse
import pandas as pd
from tensorflow.keras.models import load_model

from dataset import get_test_generator

def generate_submission(model_path, data_dir, output_file='submission.csv', batch_size=128):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print("Setting up test data generator...")
    test_generator, test_df = get_test_generator(data_dir, batch_size=batch_size)
    
    print(f"Predicting for {len(test_df)} test samples...")
    predictions = model.predict(test_generator, verbose=1)
    
    # The output is a single probability score per image
    test_df['id'] = test_df['id'].str.replace('.tif', '', regex=False)
    test_df['label'] = predictions.flatten()
    
    test_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission for Kaggle")
    parser.add_argument('--model_path', type=str, default='models/best_model.h5', help='Path to the trained model (.h5)')
    parser.add_argument('--data_dir', type=str, default='../input/histopathologic-cancer-detection/', help='Path to Kaggle dataset')
    parser.add_argument('--output_file', type=str, default='submission.csv', help='Output CSV file name')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    args = parser.parse_args()
    
    generate_submission(args.model_path, args.data_dir, args.output_file, args.batch_size)
