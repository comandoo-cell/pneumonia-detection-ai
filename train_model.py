import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

DATASET_PATH = r'C:\Users\MSI GAMING\Desktop\X-ray\chest_xray'
TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
VAL_DIR = os.path.join(DATASET_PATH, 'val')
TEST_DIR = os.path.join(DATASET_PATH, 'test')

MODEL_SAVE_PATH = r'C:\Users\MSI GAMING\Desktop\X-ray\pneumonia_mobilenetv2_NEW_TEST.h5'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

callbacks = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

history_fine = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)

print(f"\n{'='*50}")
print(f"TEST RESULTS:")
print(f"{'='*50}")
print(f"Loss:      {test_loss:.4f}")
print(f"Accuracy:  {test_accuracy*100:.2f}%")
print(f"Precision: {test_precision*100:.2f}%")
print(f"Recall:    {test_recall*100:.2f}%")
print(f"F1-Score:  {2*(test_precision*test_recall)/(test_precision+test_recall)*100:.2f}%")
print(f"{'='*50}")

all_accuracy = history.history['accuracy'] + history_fine.history['accuracy']
all_val_accuracy = history.history['val_accuracy'] + history_fine.history['val_accuracy']
all_loss = history.history['loss'] + history_fine.history['loss']
all_val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(all_accuracy, label='Train Accuracy', linewidth=2)
plt.plot(all_val_accuracy, label='Val Accuracy', linewidth=2)
plt.axvline(x=len(history.history['accuracy']), color='red', linestyle='--', label='Fine-tuning starts')
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(all_loss, label='Train Loss', linewidth=2)
plt.plot(all_val_loss, label='Val Loss', linewidth=2)
plt.axvline(x=len(history.history['loss']), color='red', linestyle='--', label='Fine-tuning starts')
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.bar(['Accuracy', 'Precision', 'Recall'], 
        [test_accuracy*100, test_precision*100, test_recall*100],
        color=['#667eea', '#11998e', '#eb3349'],
        alpha=0.8)
plt.title('Final Test Metrics', fontsize=14, fontweight='bold')
plt.ylabel('Score (%)')
plt.ylim([0, 100])
plt.grid(True, alpha=0.3, axis='y')
for i, v in enumerate([test_accuracy*100, test_precision*100, test_recall*100]):
    plt.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')

final_model_path = r'C:\Users\MSI GAMING\Desktop\X-ray\best_model_NEW_TEST.h5'
model.save(final_model_path)
