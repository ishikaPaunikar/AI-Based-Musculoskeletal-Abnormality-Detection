"""Baseline CNN and EfficientNetB0 transfer-learning model."""
from tensorflow import keras

from . import config

L = keras.layers


def build_baseline_cnn(img_size=config.IMG_SIZE, dropout=0.3, filters=(32, 64, 128, 256)):
    """Small VGG-style CNN trained from scratch. Input: float [0, 255]."""
    inputs = keras.Input((img_size, img_size, 3), name="input_image")
    x = L.Rescaling(1.0 / 255, name="rescale")(inputs)
    for i, f in enumerate(filters):
        for j in range(2):
            x = L.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}_{j}")(x)
            x = L.BatchNormalization(name=f"bn{i}_{j}")(x)
            x = L.Activation("relu", name="last_conv_act" if (i == len(filters) - 1 and j == 1) else f"relu{i}_{j}")(x)
        if i < len(filters) - 1:
            x = L.MaxPooling2D(name=f"pool{i}")(x)
    x = L.GlobalAveragePooling2D(name="head_gap")(x)
    x = L.Dropout(dropout, name="head_dropout")(x)
    out = L.Dense(1, activation="sigmoid", name="head_out")(x)
    return keras.Model(inputs, out, name="baseline_cnn")


def build_efficientnet(img_size=config.IMG_SIZE, dropout=0.3, weights="imagenet"):
    """EfficientNetB0 backbone (ImageNet) + binary head. Input: float [0, 255]
    (the backbone contains its own preprocessing)."""
    inputs = keras.Input((img_size, img_size, 3), name="input_image")
    base = keras.applications.EfficientNetB0(include_top=False, weights=weights, input_tensor=inputs)
    x = L.GlobalAveragePooling2D(name="head_gap")(base.output)
    x = L.Dropout(dropout, name="head_dropout")(x)
    out = L.Dense(1, activation="sigmoid", name="head_out")(x)
    return keras.Model(inputs, out, name="efficientnetb0_mura")


def build_model(model_type, dropout=0.3, weights="imagenet", img_size=config.IMG_SIZE):
    if weights in (None, "none", "None"):
        weights = None
    if model_type == "baseline":
        return build_baseline_cnn(img_size, dropout)
    if model_type == "efficientnet":
        return build_efficientnet(img_size, dropout, weights)
    raise ValueError(f"Unknown model type: {model_type}")


def set_backbone_trainable(model, n_unfreeze):
    """Freeze the backbone; unfreeze the last `n_unfreeze` backbone layers (BatchNorm stays frozen)."""
    backbone = [l for l in model.layers if not l.name.startswith("head_") and not isinstance(l, L.InputLayer)]
    for l in backbone:
        l.trainable = False
    if n_unfreeze > 0:
        for l in backbone[-n_unfreeze:]:
            if not isinstance(l, L.BatchNormalization):
                l.trainable = True
    for l in model.layers:
        if l.name.startswith("head_"):
            l.trainable = True


def compile_model(model, lr, optimizer="adam", weight_decay=1e-4):
    opt = (
        keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
        if optimizer == "adamw"
        else keras.optimizers.Adam(learning_rate=lr)
    )
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
