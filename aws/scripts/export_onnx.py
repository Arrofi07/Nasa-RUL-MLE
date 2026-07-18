"""
scripts/export_onnx.py
======================
Export the trained LSTM model from PyTorch (.pt) to ONNX format (.onnx).

Why ONNX?
---------
PyTorch (~800 MB unpacked) exceeds AWS Lambda's 250 MB deployment limit.
ONNX Runtime (~50 MB) is a lightweight inference engine that can run the
exported model without PyTorch installed at all. The model weights and
computation graph are serialised into a single .onnx file that ONNX Runtime
reads directly. Predictions are identical — ONNX is not a retraining, it is
a format conversion.

Size comparison
---------------
  torch==2.11.0          ~800 MB   ← too large for Lambda
  onnxruntime==1.20.0    ~50  MB   ← fits comfortably

Run this once after training:
    python scripts/export_onnx.py

Output:
    models/best_lstm.onnx        ← the exported model (deploy this to Lambda)
    models/best_lstm.onnx.check  ← numeric sanity check log
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = Path("models")
PT_PATH   = MODEL_DIR / "best_lstm.pt"
ONNX_PATH = MODEL_DIR / "best_lstm.onnx"
CFG_PATH  = MODEL_DIR / "lstm_config.json"


# ---------------------------------------------------------------------------
# Re-declare the model architecture
# ---------------------------------------------------------------------------
# We need the exact same class definition that was used during training so
# that PyTorch can reconstruct the model before exporting it.
# This mirrors src/inference/predict.py — keep them in sync.

class AttentionPooling(nn.Module):
    """Learns attention weights over time steps, returns a weighted average."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        scores  = self.score(lstm_out).squeeze(-1)             # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        return (lstm_out * weights).sum(dim=1)                 # (batch, hidden_size)


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size:    int,
        hidden_size:   int   = 64,
        num_layers:    int   = 2,
        dropout:       float = 0.2,
        bidirectional: bool  = False,
        use_attention: bool  = False,
    ):
        super().__init__()
        self.use_attention = use_attention
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = lstm_dropout,
            bidirectional = bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)

        if use_attention:
            self.attn = AttentionPooling(out_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled = self.attn(out) if self.use_attention else out[:, -1, :]
        return self.head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(
    pt_path:   Path = PT_PATH,
    onnx_path: Path = ONNX_PATH,
    cfg_path:  Path = CFG_PATH,
) -> None:
    # 1. Load config so we know the exact architecture that was trained
    with open(cfg_path) as f:
        cfg = json.load(f)

    seq_len       = cfg["seq_len"]
    hidden_size   = cfg["hidden_size"]
    num_layers    = cfg["num_layers"]
    dropout       = cfg.get("dropout", 0.2)
    bidirectional = cfg.get("bidirectional", False)
    use_attention = cfg.get("use_attention", False)

    # Read how many features the model expects from feature_cols.txt
    feature_cols_path = MODEL_DIR / "feature_cols.txt"
    with open(feature_cols_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    n_features = len(feature_cols)

    print(f"📐 Architecture:")
    print(f"   seq_len       = {seq_len}")
    print(f"   n_features    = {n_features}")
    print(f"   hidden_size   = {hidden_size}")
    print(f"   num_layers    = {num_layers}")
    print(f"   dropout       = {dropout}")
    print(f"   bidirectional = {bidirectional}")
    print(f"   use_attention = {use_attention}")

    # 2. Reconstruct the model and load weights
    # We always export on CPU — ONNX Runtime on Lambda runs on CPU anyway
    model = LSTMModel(
        input_size    = n_features,
        hidden_size   = hidden_size,
        num_layers    = num_layers,
        dropout       = dropout,
        bidirectional = bidirectional,
        use_attention = use_attention,
    )
    model.load_state_dict(
        torch.load(pt_path, map_location="cpu", weights_only=True)
    )

    # Dropout and LayerNorm behave differently during training vs inference.
    # eval() mode disables dropout and sets LayerNorm to inference behaviour.
    model.eval()

    # 3. Create a dummy input of the correct shape for the ONNX tracer.
    # torch.onnx.export traces the model by running it on this dummy tensor
    # and recording every operation. The shape must match what the model
    # will actually receive at inference time: (batch=1, seq_len, n_features).
    dummy_input = torch.randn(1, seq_len, n_features)

    # 4. Export to ONNX
    # opset_version=17 is stable and supported by onnxruntime 1.17+.
    # dynamic_axes lets the batch dimension vary at inference time so we
    # can later add batch prediction support without re-exporting.
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version    = 17,
        input_names      = ["input"],        # name used in onnxruntime session.run()
        output_names     = ["predicted_rul"],
        dynamic_axes     = {
            "input":         {0: "batch_size"},
            "predicted_rul": {0: "batch_size"},
        },
        export_params    = True,   # embed trained weights inside the .onnx file
        do_constant_folding = True, # pre-compute constant subgraphs → faster inference
    )

    pt_size   = pt_path.stat().st_size   / 1024 / 1024
    onnx_size = onnx_path.stat().st_size / 1024 / 1024
    print(f"\n✅ Exported: {onnx_path}")
    print(f"   PyTorch .pt  : {pt_size:.1f} MB")
    print(f"   ONNX .onnx   : {onnx_size:.1f} MB")

    # 5. Sanity check — run both PyTorch and ONNX on the same dummy input
    #    and assert that predictions are numerically identical (within 1e-4).
    #    If this check fails, the export went wrong somewhere.
    print("\n🔍 Sanity check: comparing PyTorch vs ONNX predictions …")

    import onnxruntime as ort  # only needed for the check, not the export itself

    # PyTorch prediction
    with torch.no_grad():
        pt_pred = model(dummy_input).numpy()

    # ONNX prediction
    session  = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_pred = session.run(
        ["predicted_rul"],
        {"input": dummy_input.numpy()},
    )[0]

    max_diff = float(np.abs(pt_pred - ort_pred).max())
    print(f"   PyTorch prediction : {pt_pred.flatten()}")
    print(f"   ONNX prediction    : {ort_pred.flatten()}")
    print(f"   Max absolute diff  : {max_diff:.2e}")

    # Write check result to a log file next to the model
    check_path = onnx_path.with_suffix(".onnx.check")
    check_path.write_text(
        f"max_diff={max_diff:.2e}\n"
        f"pt_pred={pt_pred.flatten().tolist()}\n"
        f"ort_pred={ort_pred.flatten().tolist()}\n"
    )

    assert max_diff < 1e-4, (
        f"ONNX export sanity check FAILED — max diff {max_diff:.2e} > 1e-4. "
        "Re-export and check the model architecture matches exactly."
    )
    print(f"✅ Sanity check passed (max diff {max_diff:.2e} < 1e-4)")
    print(f"   Check log → {check_path}")


if __name__ == "__main__":
    export_onnx()