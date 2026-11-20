"""Generates the two GPU notebooks (CNN versions of projects 04 and 05)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        out.append(c)
    return {"cells": out, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "accelerator": "GPU"}}


GESTURE = [
("markdown", """
# Hand gesture recognition — CNN version (GPU)

Companion to `train.py` (HOG + SVM, CPU). Trains a small CNN on the **leapGestRecog** dataset
with a **leave-people-out split** (subjects 08 and 09 are never seen during training), so the
reported accuracy reflects new users, not memorised hands.

**Run it on Kaggle:** New Notebook → *Add data* → search `leapgestrecog` → Settings → Accelerator: GPU → Run all.
On Colab, upload the dataset and change `DATA` below.
"""),
("code", """
import os, glob, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

candidates = glob.glob('/kaggle/input/**/leapGestRecog', recursive=True)
DATA = candidates[0] if candidates else 'leapGestRecog'   # <- change if needed
IMG_H, IMG_W, BATCH, SEED = 96, 256, 64, 42
GESTURES = ['palm','l','fist','fist_moved','thumb','index','ok','palm_moved','c','down']
TEST_SUBJECTS = {'08', '09'}
tf.keras.utils.set_random_seed(SEED)
print(DATA, tf.__version__, tf.config.list_physical_devices('GPU'))
"""),
("code", """
# The dataset is <subject>/<NN_gesture>/<frame>.png  — there are NO train/validation/test folders,
# so we build the split ourselves, grouped by subject.
paths, labels, subjects = [], [], []
for p in sorted(glob.glob(os.path.join(DATA, '*', '*', '*.png'))):
    subj, gest = p.split(os.sep)[-3], p.split(os.sep)[-2]
    paths.append(p); labels.append(int(gest.split('_')[0]) - 1); subjects.append(subj)
paths, labels, subjects = np.array(paths), np.array(labels), np.array(subjects)
test_mask = np.isin(subjects, list(TEST_SUBJECTS))
rng = np.random.default_rng(SEED)
train_idx = np.where(~test_mask)[0]; rng.shuffle(train_idx)
n_val = int(0.15 * len(train_idx))
val_idx, train_idx, test_idx = train_idx[:n_val], train_idx[n_val:], np.where(test_mask)[0]
print(len(train_idx), 'train /', len(val_idx), 'val /', len(test_idx), 'test (unseen people)')

def load(path, label):
    img = tf.io.decode_png(tf.io.read_file(path), channels=1)
    img = tf.image.resize(img, (IMG_H, IMG_W)) / 255.0
    return img, label

augment = tf.keras.Sequential([tf.keras.layers.RandomTranslation(0.1, 0.1),
                               tf.keras.layers.RandomZoom(0.1),
                               tf.keras.layers.RandomContrast(0.2)])
def ds(idx, train=False):
    d = tf.data.Dataset.from_tensor_slices((paths[idx], labels[idx]))
    if train: d = d.shuffle(len(idx), seed=SEED)
    d = d.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if train: d = d.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return d.batch(BATCH).prefetch(tf.data.AUTOTUNE)
train_ds, val_ds, test_ds = ds(train_idx, True), ds(val_idx), ds(test_idx)
"""),
("code", """
L = tf.keras.layers
model = tf.keras.Sequential([
    L.Input((IMG_H, IMG_W, 1)),
    L.Conv2D(32, 3, padding='same', activation='relu'), L.BatchNormalization(), L.MaxPooling2D(),
    L.Conv2D(64, 3, padding='same', activation='relu'), L.BatchNormalization(), L.MaxPooling2D(),
    L.Conv2D(128, 3, padding='same', activation='relu'), L.BatchNormalization(), L.MaxPooling2D(),
    L.Conv2D(128, 3, padding='same', activation='relu'), L.GlobalAveragePooling2D(),
    L.Dropout(0.4), L.Dense(10, activation='softmax')])
model.compile(tf.keras.optimizers.Adam(1e-3), 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
cb = [tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3)]
history = model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=cb)
"""),
("code", """
y_prob = model.predict(test_ds)
y_pred = y_prob.argmax(1); y_true = labels[test_idx]
print(f'Accuracy on UNSEEN people: {(y_pred == y_true).mean():.4f}')
print(classification_report(y_true, y_pred, target_names=GESTURES))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history['accuracy'], label='train'); ax[0].plot(history.history['val_accuracy'], label='val')
ax[0].set_title('Accuracy'); ax[0].legend()
cm = confusion_matrix(y_true, y_pred, normalize='true')
ax[1].imshow(cm, cmap='Blues'); ax[1].set_xticks(range(10), GESTURES, rotation=45); ax[1].set_yticks(range(10), GESTURES)
ax[1].set_title('Confusion matrix (unseen people)'); plt.tight_layout(); plt.savefig('cnn_results.png'); plt.show()
model.save('hand_gesture_cnn.keras')
"""),
]

FOOD = [
("markdown", """
# Food-101 dish recognition + calorie estimation — CNN (GPU)

Stage 2 of the pipeline. Fine-tunes **EfficientNetB0** (ImageNet weights) on **Food-101**
(101 dishes, 101,000 images) and maps the predicted dish to calories with `calories.py`.

**Run on Colab:** Runtime → Change runtime type → GPU → Run all (Food-101 downloads via `tensorflow_datasets`, ~5 GB).
**Run on Kaggle:** enable GPU + Internet. Upload `calories.py`, `food101_classes.py` and `assets/food_calories.csv`
next to the notebook (or clone the repo in the first cell).
"""),
("code", """
# !git clone https://github.com/<your-username>/<this-repo>.git && %cd <this-repo>/05-food-recognition-calories
import numpy as np, tensorflow as tf, tensorflow_datasets as tfds, matplotlib.pyplot as plt
from calories import lookup
IMG, BATCH, SEED = 224, 64, 42
tf.keras.utils.set_random_seed(SEED)
(train_raw, test_raw), info = tfds.load('food101', split=['train', 'validation'], as_supervised=True, with_info=True)
CLASSES = info.features['label'].names
print(len(CLASSES), 'classes;', info.splits['train'].num_examples, 'train /', info.splits['validation'].num_examples, 'test')
"""),
("code", """
def prep(img, label):
    return tf.image.resize(img, (IMG, IMG)), label   # EfficientNet does its own normalisation
aug = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'), tf.keras.layers.RandomRotation(0.1),
                           tf.keras.layers.RandomZoom(0.15)])
train_ds = (train_raw.map(prep, num_parallel_calls=tf.data.AUTOTUNE).shuffle(4096, seed=SEED)
            .batch(BATCH).map(lambda x, y: (aug(x, training=True), y)).prefetch(tf.data.AUTOTUNE))
test_ds = test_raw.map(prep, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)
"""),
("code", """
base = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(IMG, IMG, 3))
base.trainable = False
inp = tf.keras.Input((IMG, IMG, 3))
x = base(inp, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)
model = tf.keras.Model(inp, out)

# phase 1: train the new head
model.compile(tf.keras.optimizers.Adam(1e-3), 'sparse_categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name='top5')])
h1 = model.fit(train_ds, validation_data=test_ds, epochs=3)

# phase 2: unfreeze the top of the backbone and fine-tune with a small learning rate
base.trainable = True
for layer in base.layers[:-60]:
    layer.trainable = False
model.compile(tf.keras.optimizers.Adam(1e-4), 'sparse_categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name='top5')])
h2 = model.fit(train_ds, validation_data=test_ds, epochs=5,
               callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
loss, acc, top5 = model.evaluate(test_ds)
print(f'Food-101 test accuracy: top-1 {acc:.4f}, top-5 {top5:.4f}')
model.save('food101_efficientnetb0.keras')
"""),
("code", """
def recognise_and_estimate(image_path, top_k=3):
    img = tf.image.resize(tf.io.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False), (IMG, IMG))
    probs = model.predict(img[None], verbose=0)[0]
    out = []
    for i in probs.argsort()[::-1][:top_k]:
        cal = lookup(CLASSES[i])
        out.append({'dish': CLASSES[i], 'confidence': float(probs[i]),
                    'calories': f"{cal['kcal_per_serving']:.0f} kcal per {cal['serving']}" if cal else 'not in table'})
    return out

# try it on any food photo you upload:
# recognise_and_estimate('my_lunch.jpg')
"""),
]

for path, cells in [(ROOT / "04-hand-gesture-recognition/gesture_cnn_gpu.ipynb", GESTURE),
                    (ROOT / "05-food-recognition-calories/food101_cnn_gpu.ipynb", FOOD)]:
    path.write_text(json.dumps(nb(cells), indent=1))
    print("wrote", path)
