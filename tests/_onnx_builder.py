"""
A minimal ONNX (protobuf) serializer, for TESTS ONLY.

The `onnx` python package isn't a runtime dependency of this project, but we
still want the vision serving path exercised against a *real* onnxruntime
session rather than a mock. So this writes a tiny, valid ONNX file by hand.

It builds a stand-in model with the same output contract as the real exported
classifier:
    input:  "input"                  float32 [N,3,224,224]   (values in [0,1])
    output: "embedding"              float32 [N,D]
    output: "logits_<task>"          float32 [N,C_task]      (one per task)

It also reproduces the baked-in normalisation (Sub/Div) that
`ml/export_onnx.py` inserts, so tests can verify the graph normalises exactly
like `vision_preprocess.normalize_reference`.

This is not a general ONNX writer — just enough proto to be valid.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

# ─── protobuf wire primitives ─────────────────────────────────────────
def _varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _bytes_field(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def _str_field(field: int, s: str) -> bytes:
    return _bytes_field(field, s.encode())


def _int_field(field: int, v: int) -> bytes:
    return _tag(field, 0) + _varint(v)


def _packed_ints(field: int, vals: list[int]) -> bytes:
    return _bytes_field(field, b"".join(_varint(v) for v in vals))


# ─── ONNX messages ────────────────────────────────────────────────────
def _tensor_proto(name: str, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    body = _packed_ints(1, list(arr.shape))      # dims
    body += _int_field(2, 1)                     # data_type = FLOAT
    body += _str_field(8, name)                  # name
    body += _bytes_field(9, arr.tobytes())       # raw_data (little-endian f32)
    return body


def _value_info(name: str, shape: list) -> bytes:
    dims = b""
    for d in shape:
        dims += _bytes_field(1, _str_field(2, d) if isinstance(d, str) else _int_field(1, d))
    tensor_type = _int_field(1, 1) + _bytes_field(2, dims)   # elem_type=FLOAT, shape
    type_proto = _bytes_field(1, tensor_type)                # TypeProto.tensor_type
    return _str_field(1, name) + _bytes_field(2, type_proto)


def _attr_ints(name: str, vals: list[int]) -> bytes:
    return _str_field(1, name) + _packed_ints(8, vals) + _int_field(20, 7)  # type=INTS


def _attr_int(name: str, val: int) -> bytes:
    return _str_field(1, name) + _int_field(3, val) + _int_field(20, 2)     # type=INT


def _node(op: str, inputs: list[str], outputs: list[str], name: str, attrs: bytes = b"") -> bytes:
    body = b"".join(_str_field(1, i) for i in inputs)
    body += b"".join(_str_field(2, o) for o in outputs)
    body += _str_field(3, name) + _str_field(4, op)
    if attrs:
        body += _bytes_field(5, attrs)
    return body


def build_fake_model(
    out_dir: Path,
    tasks: dict[str, list[str]],
    embedding_dim: int = 32,
    input_size: int = 224,
    seed: int = 7,
) -> tuple[Path, Path]:
    """Write a valid ONNX model + labels.json mirroring the real export contract.

    Graph:
        input -> Sub(mean) -> Div(std)        [baked-in normalisation]
              -> ReduceMean(axes=2,3)         [global average pool -> N,3]
              -> MatMul(W_emb)                [-> N,D]  = "embedding"
              -> MatMul(W_task) per task      [-> N,C]  = "logits_<task>"
    """
    from app.vision_preprocess import IMAGENET_MEAN, IMAGENET_STD

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    w_emb = rng.normal(0, 0.5, size=(3, embedding_dim)).astype(np.float32)
    heads = {t: rng.normal(0, 0.5, size=(embedding_dim, len(l))).astype(np.float32)
             for t, l in tasks.items()}

    inits = [_tensor_proto("mean", mean), _tensor_proto("std", std),
             _tensor_proto("W_emb", w_emb)]
    inits += [_tensor_proto(f"W_{t}", w) for t, w in heads.items()]

    nodes = [
        _node("Sub", ["input", "mean"], ["centered"], "sub_mean"),
        _node("Div", ["centered", "std"], ["normalized"], "div_std"),
        # NOTE: we deliberately do NOT emit `keepdims=0`. In proto3 a scalar 0 is
        # indistinguishable from "unset", so ONNX rejects it as a type/data
        # mismatch. Use the default (keepdims=1) then Flatten, which takes no
        # attributes at all: [N,3,1,1] -> [N,3].
        _node("ReduceMean", ["normalized"], ["gap4d"], "gap", _attr_ints("axes", [2, 3])),
        _node("Flatten", ["gap4d"], ["gap"], "flatten"),
        _node("MatMul", ["gap", "W_emb"], ["embedding"], "proj_emb"),
    ]
    nodes += [_node("MatMul", ["embedding", f"W_{t}"], [f"logits_{t}"], f"head_{t}")
              for t in tasks]

    graph = b"".join(_bytes_field(1, n) for n in nodes)
    graph += _str_field(2, "fashion_classifier_stub")
    graph += b"".join(_bytes_field(5, i) for i in inits)
    graph += _bytes_field(11, _value_info("input", ["batch", 3, input_size, input_size]))
    graph += _bytes_field(12, _value_info("embedding", ["batch", embedding_dim]))
    graph += b"".join(_bytes_field(12, _value_info(f"logits_{t}", ["batch", len(l)]))
                      for t, l in tasks.items())

    opset = _str_field(1, "") + _int_field(2, 17)
    model = _int_field(1, 8)                       # ir_version
    model += _str_field(2, "zintoo-test-builder")  # producer_name
    model += _bytes_field(7, graph)
    model += _bytes_field(8, opset)

    model_path = out_dir / "fashion_classifier.onnx"
    labels_path = out_dir / "labels.json"
    model_path.write_bytes(model)
    labels_path.write_text(json.dumps({
        "tasks": tasks,
        "embedding_dim": embedding_dim,
        "input_size": input_size,
        "trained_at": "1970-01-01T00:00:00Z",
        "metrics": {"note": "stub model for tests — not trained"},
        "preprocess": {"resize_short": 256, "crop": input_size,
                       "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
    }, indent=2))
    return model_path, labels_path
