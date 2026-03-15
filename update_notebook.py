import json
import os

notebook_path = '/Users/josh/Projects/cnn_cancer_detetiion_uni_project/histopathologic_cancer_detection.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_training_code = [
    "# Actual training pipeline\n",
    "from sklearn.model_selection import train_test_split\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint\n",
    "\n",
    "def train_model(df_labels, data_dir='../input/histopathologic-cancer-detection/'):\n",
    "    print(\"Setting up data generators...\")\n",
    "    # Copy df to avoid modifying original\n",
    "    df = df_labels.copy()\n",
    "    df['id'] = df['id'] + '.tif'\n",
    "    df['label'] = df['label'].astype(str)\n",
    "    \n",
    "    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])\n",
    "    \n",
    "    train_dir = os.path.join(data_dir, 'train')\n",
    "    if not os.path.exists(train_dir):\n",
    "        print(f\"Warning: {train_dir} not found. Skipping generator creation.\")\n",
    "        return None\n",
    "    \n",
    "    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)\n",
    "    val_datagen = ImageDataGenerator(rescale=1./255)\n",
    "    \n",
    "    train_generator = train_datagen.flow_from_dataframe(\n",
    "        dataframe=train_df, directory=train_dir, x_col='id', y_col='label',\n",
    "        target_size=(96, 96), batch_size=128, class_mode='binary'\n",
    "    )\n",
    "    \n",
    "    val_generator = val_datagen.flow_from_dataframe(\n",
    "        dataframe=val_df, directory=train_dir, x_col='id', y_col='label',\n",
    "        target_size=(96, 96), batch_size=128, class_mode='binary'\n",
    "    )\n",
    "    \n",
    "    print(\"Setting up callbacks...\")\n",
    "    callbacks = [\n",
    "        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max'),\n",
    "        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, mode='max'),\n",
    "        ModelCheckpoint('best_model.h5', monitor='val_auc', save_best_only=True, mode='max', verbose=1)\n",
    "    ]\n",
    "    \n",
    "    print(\"Starting training...\")\n",
    "    history = model.fit(\n",
    "        train_generator,\n",
    "        validation_data=val_generator,\n",
    "        epochs=15,\n",
    "        callbacks=callbacks\n",
    "    )\n",
    "    \n",
    "    print(\"Plotting results...\")\n",
    "    plt.figure(figsize=(12, 4))\n",
    "    plt.subplot(1, 2, 1)\n",
    "    plt.plot(history.history['loss'], label='Train Loss')\n",
    "    plt.plot(history.history['val_loss'], label='Val Loss')\n",
    "    plt.title('Loss Evolution')\n",
    "    plt.legend()\n",
    "    \n",
    "    plt.subplot(1, 2, 2)\n",
    "    plt.plot(history.history['auc'], label='Train AUC')\n",
    "    plt.plot(history.history['val_auc'], label='Val AUC')\n",
    "    plt.title('AUC Evolution')\n",
    "    plt.legend()\n",
    "    plt.show()\n",
    "    \n",
    "    return history\n",
    "\n",
    "if 'df' in locals() and os.path.exists(TRAIN_DIR):\n",
    "    print(\"Executing train_model()\")\n",
    "    history = train_model(df, DATA_DIR)\n",
    "else:\n",
    "    print(\"Dataset not found locally. Code ready for Kaggle execution.\")\n"
]

prediction_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Inference and Submission\n",
        "import glob\n",
        "def generate_submission(model, data_dir='../input/histopathologic-cancer-detection/'):\n",
        "    test_dir = os.path.join(data_dir, 'test')\n",
        "    if not os.path.exists(test_dir):\n",
        "        print(\"Test directory not found. Skipping submission generation.\")\n",
        "        return\n",
        "    \n",
        "    print(\"Generating submission...\")\n",
        "    test_files = glob.glob(os.path.join(test_dir, '*.tif'))\n",
        "    test_df = pd.DataFrame({'id': [os.path.basename(f) for f in test_files]})\n",
        "    \n",
        "    test_datagen = ImageDataGenerator(rescale=1./255)\n",
        "    test_generator = test_datagen.flow_from_dataframe(\n",
        "        dataframe=test_df,\n",
        "        directory=test_dir,\n",
        "        x_col='id',\n",
        "        y_col=None,\n",
        "        target_size=(96, 96),\n",
        "        batch_size=128,\n",
        "        class_mode=None,\n",
        "        shuffle=False\n",
        "    )\n",
        "    \n",
        "    predictions = model.predict(test_generator, verbose=1)\n",
        "    \n",
        "    test_df['id'] = test_df['id'].str.replace('.tif', '', regex=False)\n",
        "    test_df['label'] = predictions.flatten()\n",
        "    \n",
        "    test_df.to_csv('submission.csv', index=False)\n",
        "    print(\"Submission saved to submission.csv\")\n",
        "\n",
        "if 'model' in locals():\n",
        "    generate_submission(model, DATA_DIR)\n"
    ]
}

# Find the cell containing the pseudo-code logic and replace it
cell_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any("Pseudo-code for training pipeline" in line for line in cell.get('source', [])):
        cell_index = i
        break

if cell_index != -1:
    nb['cells'][cell_index]['source'] = new_training_code
    
    # Insert prediction cell right after the training cell
    nb['cells'].insert(cell_index + 1, prediction_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print("Notebook updated successfully.")
else:
    print("Could not find the target cell to replace.")
