import uuid
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import logging

MAX_WORDS = 20000
MAX_LEN = 500
MODELS_DIR = Path("models_dl")
MODELS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("training")
logging.basicConfig(level=logging.INFO)

def _save_tokenizer(tok: Tokenizer, path: Path):
    path.write_text(tok.to_json(), encoding='utf-8')

def _load_tokenizer(path: Path) -> Tokenizer:
    tok_json = path.read_text(encoding='utf-8')
    return tokenizer_from_json(tok_json)

def _create_model(vocab_size: int, seq_len: int, lr: float) -> tf.keras.Model:
    model = models.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=96, input_length=seq_len),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

class EpochLogger(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = (
            f"Epoch {epoch+1} - "
            f"loss: {logs.get('loss', 0):.4f}, "
            f"acc: {logs.get('accuracy', 0):.4f}, "
            f"val_loss: {logs.get('val_loss', 0):.4f}, "
            f"val_accuracy: {logs.get('val_accuracy', 0):.4f}"
        )
        logger.info(msg)

class TextGenerator(tf.keras.utils.Sequence):
    def __init__(self, texts, labels, tokenizer, max_len, batch_size, shuffle=True):
        self.texts = np.asarray(texts)
        self.labels = np.asarray(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.texts) / self.batch_size))

    def __getitem__(self, idx):
        batch_texts = self.texts[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_labels = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]
        seqs = self.tokenizer.texts_to_sequences(batch_texts)
        X = pad_sequences(seqs, maxlen=self.max_len, dtype="int32")
        y = to_categorical(batch_labels, num_classes=3)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            idxs = np.arange(len(self.texts))
            np.random.shuffle(idxs)
            self.texts = self.texts[idxs]
            self.labels = self.labels[idxs]

def train_new(df: pd.DataFrame, lr: float, batch_size: int, epochs: int, patience: int):
    # Prepare data
    df = df.dropna(subset=['text', 'target']).copy()
    df['target'] = df['target'].astype(int) + 1
    df['text'] = df['text'].astype(str)

    X = df['text'].values
    y = df['target'].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y))>1 else None
    )

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    train_gen = TextGenerator(X_train, y_train, tokenizer, MAX_LEN, batch_size, shuffle=True)
    val_gen = TextGenerator(X_val, y_val, tokenizer, MAX_LEN, batch_size, shuffle=False)

    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(cw))

    model = _create_model(MAX_WORDS, MAX_LEN, lr)
    cbs = [
        EpochLogger(),
        callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)
    ]
    print(">>> training started")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=0
    )
    print(">>> training finished")

    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

    y_true, y_pred = [], []
    for i in range(len(val_gen)):
        xb, yb = val_gen[i]
        y_true.extend(yb)
        y_pred.extend(model.predict(xb, verbose=0))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_bin = label_binarize(np.argmax(y_true, axis=1), classes=[0, 1, 2])
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    metrics = {
        'accuracy': float(accuracy_score(np.argmax(y_true, 1), np.argmax(y_pred, 1))),
        'f1_weighted': float(f1_score(np.argmax(y_true, 1), np.argmax(y_pred, 1), average='weighted')),
        'auc_macro': float(np.mean(list(roc_auc.values()))),
        'auc_per_class': {i: float(v) for i, v in roc_auc.items()}
    }

    model_id = f"dl_{uuid.uuid4().hex[:8]}"
    model.save(MODELS_DIR / f"{model_id}_model.keras", include_optimizer=False)
    _save_tokenizer(tokenizer, MODELS_DIR / f"{model_id}_tokenizer.json")
    (MODELS_DIR / f"{model_id}_meta.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    return {"model_id": model_id, "model": model, "tokenizer": tokenizer, "metrics": metrics}

def finetune(model, tokenizer, df_new: pd.DataFrame, learning_rate: float, batch_size: int, epochs: int, patience: int):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras import callbacks, optimizers

    df_new = df_new.dropna(subset=['text', 'target'])[['text', 'target']]
    if len(df_new) < 5:
        raise ValueError("Для дообучения нужно ≥ 5 строк.")
    df_new['target'] = df_new['target'].astype(int) + 1

    y = df_new['target'].values
    X = df_new['text'].values
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    batch_size = min(batch_size, len(X_tr), len(X_val))

    train_gen = TextGenerator(X_tr, y_tr, tokenizer, MAX_LEN, batch_size, shuffle=True)
    val_gen   = TextGenerator(X_val, y_val, tokenizer, MAX_LEN, batch_size, shuffle=False)

    model.compile(optimizer=optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        train_gen, validation_data=val_gen,
        epochs=epochs,
        callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)],
        verbose=0
    )

    from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_true, y_pred = [], []
    for i in range(len(val_gen)):
        Xb, yb = val_gen[i]
        if Xb.shape[0] == 0:
            continue
        preds = model.predict(Xb, verbose=0)
        y_true.extend(yb)
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(np.argmax(y_true, 1), np.argmax(y_pred, 1))
    f1 = f1_score(np.argmax(y_true, 1), np.argmax(y_pred, 1), average='weighted')

    y_bin = label_binarize(np.argmax(y_true, axis=1), classes=[0, 1, 2])
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    auc_macro = float(np.mean(list(roc_auc.values())))

    return {
        "metrics": {
            "accuracy": float(acc),
            "f1_weighted": float(f1),
            "auc_macro": auc_macro,
            "auc_per_class": {str(i): float(v) for i, v in roc_auc.items()}
        }
    }

def predict(model, tokenizer, texts):
    seqs = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_LEN, dtype="int32")
    probs = model.predict(seqs, verbose=0)
    return (np.argmax(probs, axis=1) - 1).tolist()

def load_by_id(mid: str):
    m = tf.keras.models.load_model(MODELS_DIR / f"{mid}_model.keras", compile=False)
    tok = _load_tokenizer(MODELS_DIR / f"{mid}_tokenizer.json")
    return m, tok
