import os, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Fix randomness for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

IMG_SIZE, BATCH_SIZE, NUM_CLASSES = (224, 224), 64 , 2
paths = {s: f'dataset/{s}' for s in ['train','validation','test']}

def create_gens():
    """
    Create image generators for training, validation, and test sets.
    Includes data augmentation for training.
    """
    train_gen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode='nearest'
    ).flow_from_directory(
        directory=paths['train'],
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_test_gen = ImageDataGenerator(rescale=1./255)

    val_gen = val_test_gen.flow_from_directory(
        directory=paths['validation'],
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_gen = val_test_gen.flow_from_directory(
        directory=paths['test'],
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen

def build_model():
    """
    Build MobileNetV2-based model with transfer learning.
    Top layers: GAP + Dense + Dropout + BatchNorm + Softmax.
    """
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE,3))
    base.trainable = False
    model = Sequential([
        base, 
        layers.GlobalAveragePooling2D(), 
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'), 
        layers.BatchNormalization(),
        layers.Dropout(0.5), 
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(Adam(1e-4), 'categorical_crossentropy', ['accuracy'])
    return model

def train(model, train_gen, val_gen, epochs=30, fine_tune_epochs=15):
    """
    Train the model in two phases:
      1. Train only top layers (base frozen)
      2. Fine-tune last layers of base model
    """
    cb = [
        tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-4)
    ]
    
    steps = train_gen.samples // BATCH_SIZE
    val_steps = val_gen.samples // BATCH_SIZE

    # Phase 1: Train classifier head
    h1 = model.fit(train_gen, epochs=epochs, validation_data=val_gen,
                   steps_per_epoch=steps, validation_steps=val_steps, callbacks=cb, verbose=1)

    # Phase 2: Fine-tune base model (partial unfreeze)
    model.layers[0].trainable = True
    for l in model.layers[0].layers[:100]:
        l.trainable = False

    model.compile(Adam(1e-5), 'categorical_crossentropy', ['accuracy'])

    h2 = model.fit(train_gen, epochs=epochs+fine_tune_epochs, initial_epoch=h1.epoch[-1],
                   validation_data=val_gen, steps_per_epoch=steps, validation_steps=val_steps, callbacks=cb, verbose=1)

    # Merge histories
    return model, {k: h1.history[k] + h2.history[k] for k in h1.history}

def evaluate(model, test_gen):
    """
    Evaluate best saved model on the test set.
    Generate classification report and confusion matrix.
    """
    model = tf.keras.models.load_model('models/best_model.h5')
    loss, acc = model.evaluate(test_gen, steps=test_gen.samples // BATCH_SIZE, verbose=1)
    test_gen.reset()
    preds = model.predict(test_gen, steps=test_gen.samples // BATCH_SIZE, verbose=1)
    y_true, y_pred = test_gen.classes[:len(preds)], preds.argmax(axis=1)
    names = list(test_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=names)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300)
    plt.close()
    print(report)
    return acc, loss

def convert_tflite(model, val_gen):
    """
    Convert trained Keras model to quantized TFLite format
    with representative dataset for int8 optimization.
    """
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    val_iter = iter(val_gen)
    def rep_data_gen():
        for _ in range(100):
            yield [next(val_iter)[0].astype(np.float32)]
    conv.representative_dataset = rep_data_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    tflite_model = conv.convert()
    with open('models/model.tflite', 'wb') as f: 
        f.write(tflite_model)

# Prepare folders for outputs
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Run pipeline
train_gen, val_gen, test_gen = create_gens()
model = build_model()
model, history = train(model, train_gen, val_gen)
acc, loss = evaluate(model, test_gen)
convert_tflite(model, val_gen)
