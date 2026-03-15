import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir, batch_size=128, img_size=(96, 96), val_split=0.15):
    """
    Creates and returns train and validation data generators.
    """
    train_dir = os.path.join(data_dir, 'train')
    labels_file = os.path.join(data_dir, 'train_labels.csv')
    
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels CSV not found at {labels_file}")
        
    df = pd.read_csv(labels_file)
    df['id'] = df['id'] + '.tif'
    df['label'] = df['label'].astype(str)
    
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=42, stratify=df['label'])
    
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col='id',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=train_dir,
        x_col='id',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    return train_generator, val_generator, train_df, val_df


def get_test_generator(data_dir, batch_size=128, img_size=(96, 96)):
    """
    Creates and returns a test data generator for inference without shuffling.
    """
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at {test_dir}")
        
    import glob
    test_files = glob.glob(os.path.join(test_dir, '*.tif'))
    test_df = pd.DataFrame({'id': [os.path.basename(f) for f in test_files]})
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=test_dir,
        x_col='id',
        y_col=None,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    
    return test_generator, test_df
