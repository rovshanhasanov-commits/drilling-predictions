"""Seq2seq encoder-decoder model with cross-attention + per-target output heads.

Ported from seq2seq_ss.ipynb with two adjustments:
  1. Target naming uses the 4-level hierarchy ('phase', 'phase_step', ...) rather than
     '*_next' — because target shifting now lives in training/data.py (single shift).
  2. n_classes already contains the EOO extra class, so output head widths are correct
     without explicit +1 in this file.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers

HIERARCHY = ["phase", "phase_step", "major_ops_code", "operation"]


def build_seq2seq_model(
    emb_strategy: str,
    cat_input_cols: list[str],
    cat_encoders: dict,
    n_classes: dict,
    n_numeric: int,
    seq_len: int,
    n_future: int,
    enc_lstm_units: list[int],
    dec_lstm_units: int,
    dense_units: list[int],
    dropout: float,
    learning_rate: float,
    loss_weights: dict,
    active_targets: list[str],
    predict_duration: bool,
    dec_target_edims: dict,
):
    """Returns (training_model, encoder_model, decoder_step_model)."""

    ENC_SEPARATE_EDIMS = {
        "phase_enc": 4, "phase_step_enc": 8,
        "major_ops_code_enc": 16, "operation_enc": 32,
    }

    # ===== ENCODER =====
    enc_inputs: list = []
    enc_merge: list = []

    if emb_strategy == "dummies":
        ni = keras.Input(shape=(seq_len, n_numeric), name="numeric_input")
        enc_inputs.append(ni); enc_merge.append(ni)
    elif emb_strategy == "embed_separate":
        for col in cat_input_cols:
            raw = col.replace("_enc", "")
            # +1 so the embedding can index the start-token id used by the decoder
            n_v = len(cat_encoders[raw].classes_) + 1
            ci = keras.Input(shape=(seq_len,), name=col + "_input", dtype="int32")
            emb = layers.Embedding(n_v, ENC_SEPARATE_EDIMS.get(col, 16), name=f"enc_{col}_emb")(ci)
            enc_inputs.append(ci); enc_merge.append(emb)
        ni = keras.Input(shape=(seq_len, n_numeric), name="numeric_input")
        enc_inputs.append(ni); enc_merge.append(ni)
    elif emb_strategy == "embed_state":
        n_s = len(cat_encoders["state"].classes_) + 1
        si = keras.Input(shape=(seq_len,), name="state_enc_input", dtype="int32")
        emb = layers.Embedding(n_s, 64, name="enc_state_emb")(si)
        enc_inputs.append(si); enc_merge.append(emb)
        ni = keras.Input(shape=(seq_len, n_numeric), name="numeric_input")
        enc_inputs.append(ni); enc_merge.append(ni)
    else:
        raise ValueError(f"Unknown embedding strategy {emb_strategy!r}")

    x = layers.Concatenate(axis=-1)(enc_merge) if len(enc_merge) > 1 else enc_merge[0]
    x = layers.BatchNormalization(name="enc_bn")(x)

    enc_h = enc_c = None
    for i, u in enumerate(enc_lstm_units):
        return_state = (i == len(enc_lstm_units) - 1)
        lstm = layers.LSTM(u, return_sequences=True, return_state=return_state, name=f"enc_lstm_{i+1}")
        if return_state:
            x, enc_h, enc_c = lstm(x)
        else:
            x = lstm(x)
        x = layers.Dropout(dropout, name=f"enc_drop_{i+1}")(x)
    enc_outputs = x

    # ===== DECODER =====
    dec_inputs: list = []
    dec_parts: list = []
    dec_emb_layers: dict = {}

    for t in active_targets:
        di = keras.Input(shape=(n_future,), name=f"dec_{t}_input", dtype="int32")
        dec_inputs.append(di)
        edim = dec_target_edims.get(f"{t}_next", dec_target_edims.get(t, 16))
        emb = layers.Embedding(n_classes[t] + 1, edim, name=f"dec_{t}_emb")
        dec_emb_layers[t] = emb
        dec_parts.append(emb(di))

    if predict_duration:
        dur_in = keras.Input(shape=(n_future,), name="dec_duration_input")
        dec_inputs.append(dur_in)
        dur_expanded = layers.Reshape((n_future, 1), name="dec_dur_reshape")(dur_in)
        dec_parts.append(dur_expanded)

    dec_emb = layers.Concatenate(axis=-1, name="dec_emb_concat")(dec_parts) if len(dec_parts) > 1 else dec_parts[0]

    if enc_lstm_units[-1] != dec_lstm_units:
        enc_h = layers.Dense(dec_lstm_units, name="enc_h_proj")(enc_h)
        enc_c = layers.Dense(dec_lstm_units, name="enc_c_proj")(enc_c)

    dec_lstm = layers.LSTM(dec_lstm_units, return_sequences=True, return_state=True, name="dec_lstm")
    dec_out, _, _ = dec_lstm(dec_emb, initial_state=[enc_h, enc_c])

    if enc_lstm_units[-1] != dec_lstm_units:
        enc_outputs_proj = layers.Dense(dec_lstm_units, name="enc_out_proj")(enc_outputs)
    else:
        enc_outputs_proj = enc_outputs

    cross_attn = layers.Attention(name="cross_attention")
    attn_ctx = cross_attn([dec_out, enc_outputs_proj])

    combined = layers.Concatenate(axis=-1, name="dec_attn_concat")([dec_out, attn_ctx])

    sx = combined
    for i, u in enumerate(dense_units):
        sx = layers.Dense(u, activation="relu", name=f"dec_dense_{i+1}")(sx)
        sx = layers.Dropout(dropout, name=f"dec_ddrop_{i+1}")(sx)

    # Heads
    outputs: dict = {}
    for t in active_targets:
        outputs[t] = layers.Dense(n_classes[t], activation="softmax", name=t)(sx)
    if predict_duration:
        dur_dense = layers.Dense(1, activation="linear", name="dur_linear")(sx)
        outputs["duration"] = layers.Lambda(lambda z: z[..., 0], name="duration")(dur_dense)

    training_model = Model(inputs=enc_inputs + dec_inputs, outputs=outputs)

    losses, metrics = {}, {}
    for t in active_targets:
        losses[t] = "sparse_categorical_crossentropy"
        metrics[t] = "accuracy"
    if predict_duration:
        losses["duration"] = "huber"
        metrics["duration"] = "mae"

    lw = {t: loss_weights[t] for t in active_targets}
    if predict_duration:
        lw["duration"] = loss_weights.get("duration_next", loss_weights.get("duration", 1.0))

    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=lw,
        metrics=metrics,
    )

    # ===== ENCODER (inference) =====
    encoder_model = Model(inputs=enc_inputs, outputs=[enc_outputs_proj, enc_h, enc_c], name="encoder")

    # ===== DECODER STEP (inference) =====
    dec_step_inputs: list = []
    dec_step_parts: list = []
    for t in active_targets:
        si = keras.Input(shape=(1,), name=f"step_{t}_input", dtype="int32")
        dec_step_inputs.append(si)
        dec_step_parts.append(dec_emb_layers[t](si))
    if predict_duration:
        sdi = keras.Input(shape=(1,), name="step_dur_input")
        dec_step_inputs.append(sdi)
        dec_step_parts.append(layers.Reshape((1, 1))(sdi))
    step_emb = layers.Concatenate(axis=-1)(dec_step_parts) if len(dec_step_parts) > 1 else dec_step_parts[0]

    enc_out_in = keras.Input(shape=(seq_len, dec_lstm_units), name="step_enc_out")
    h_in = keras.Input(shape=(dec_lstm_units,), name="step_h_in")
    c_in = keras.Input(shape=(dec_lstm_units,), name="step_c_in")

    step_dec_out, step_h, step_c = dec_lstm(step_emb, initial_state=[h_in, c_in])
    step_attn = cross_attn([step_dec_out, enc_out_in])
    step_combined = layers.Concatenate(axis=-1)([step_dec_out, step_attn])

    sx = step_combined
    for i in range(len(dense_units)):
        sx = training_model.get_layer(f"dec_dense_{i+1}")(sx)
        sx = training_model.get_layer(f"dec_ddrop_{i+1}")(sx)

    step_outputs = [training_model.get_layer(t)(sx) for t in active_targets]
    if predict_duration:
        step_outputs.append(training_model.get_layer("dur_linear")(sx))
    step_outputs.extend([step_h, step_c])

    decoder_step_model = Model(
        inputs=dec_step_inputs + [enc_out_in, h_in, c_in],
        outputs=step_outputs,
        name="decoder_step",
    )

    return training_model, encoder_model, decoder_step_model
