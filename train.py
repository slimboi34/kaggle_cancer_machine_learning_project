import os
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from model import build_dual_path_model
from dataset import get_data_generators

def train(data_dir, epochs=15, batch_size=128, save_dir='models'):
    print(f"Data directory: {data_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Setting up data generators...")
    train_generator, val_generator, _, _ = get_data_generators(data_dir, batch_size=batch_size)
    
    print("Building model...")
    model = build_dual_path_model()
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max'),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, mode='max'),
        ModelCheckpoint(os.path.join(save_dir, 'best_model.h5'), monitor='val_auc', save_best_only=True, mode='max', verbose=1)
    ]
    
    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    
    print("Plotting results...")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Evolution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.title('AUC Evolution')
    plt.legend()
    plt.savefig('training_history.png')
    
    # Save the final model just in case
    model.save(os.path.join(save_dir, 'final_model.h5'))
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the cancer detection model")
    parser.add_argument('--data_dir', type=str, default='../input/histopathologic-cancer-detection/', help='Path to Kaggle dataset')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    train(args.data_dir, args.epochs, args.batch_size, args.save_dir)
