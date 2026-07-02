#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import cv2
import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image

from models.modeling_bridge_wa import BridgeWA
from models.processing_vision_action import VisionActionProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Deploy BridgeWA as a Libero HTTP policy server")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8018)
    parser.add_argument("--advertise_host", type=str, default="127.0.0.1")
    parser.add_argument("--connection_info", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--default_steps", type=int, default=10)
    parser.add_argument("--default_domain_id", type=int, default=3)
    parser.add_argument(
        "--image_layout",
        type=str,
        default="default",
        choices=("default", "front_wrist_composite"),
        help="Use front + side-by-side wrist image layout for domains trained with a composed wrist view.",
    )
    return parser.parse_args()


def _decode_image(value: Any) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json_numpy.loads(value)
        except Exception:
            if os.path.exists(value):
                return Image.open(value).convert("RGB")
            raise
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            decoded = cv2.imdecode(value, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("Failed to decode image bytes.")
            value = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return Image.fromarray(value.astype(np.uint8)).convert("RGB")
    if isinstance(value, (list, tuple)):
        return Image.fromarray(np.asarray(value).astype(np.uint8)).convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    return Image.fromarray(np.asarray(value).astype(np.uint8)).convert("RGB")


def _to_model_tensor(value: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if value.is_floating_point():
        return value.to(device=device, dtype=dtype)
    return value.to(device=device)


def _compose_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    left = left.convert("RGB")
    right = right.convert("RGB")
    if right.size[1] != left.size[1]:
        scale = left.size[1] / max(1, right.size[1])
        right = right.resize((max(1, int(right.size[0] * scale)), left.size[1]))
    canvas = Image.new("RGB", (left.size[0] + right.size[0], left.size[1]))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.size[0], 0))
    return canvas


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _write_connection_info(path: Path, *, host: str, port: int, model_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"host": host, "port": port, "model_path": model_path}, f, ensure_ascii=False, indent=2)


def build_app(
    model: BridgeWA,
    processor: VisionActionProcessor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    default_steps: int,
    default_domain_id: int,
    image_layout: str,
) -> FastAPI:
    app = FastAPI()
    inference_lock = threading.Lock()

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/act")
    def act(payload: dict[str, Any]):
        try:
            model.eval()

            images = []
            if image_layout == "front_wrist_composite":
                front = _decode_image(payload.get("image0"))
                left_wrist = _decode_image(payload.get("image1"))
                right_wrist = _decode_image(payload.get("image2"))
                if front is not None:
                    images.append(front)
                if left_wrist is not None and right_wrist is not None:
                    images.append(_compose_side_by_side(left_wrist, right_wrist))
            else:
                for key in ("image0", "image1", "image2"):
                    if key in payload:
                        image = _decode_image(payload[key])
                        if image is not None:
                            images.append(image)
            if not images:
                return JSONResponse({"error": "No valid images found."}, status_code=400)

            language_instruction = payload.get("language_instruction")
            if not language_instruction:
                return JSONResponse({"error": "language_instruction is required."}, status_code=400)

            inputs = processor(images=images, language_instruction=language_instruction)
            proprio_raw = payload.get("proprio")
            if proprio_raw is None:
                return JSONResponse({"error": "proprio is required."}, status_code=400)
            proprio = torch.as_tensor(np.asarray(json_numpy.loads(proprio_raw))).unsqueeze(0)
            domain_id = torch.tensor([int(payload.get("domain_id", default_domain_id))], dtype=torch.long)

            inputs = {k: _to_model_tensor(v, device=device, dtype=dtype) for k, v in inputs.items()}
            inputs["proprio"] = _to_model_tensor(proprio, device=device, dtype=dtype)
            inputs["domain_id"] = domain_id.to(device=device)

            steps = int(payload.get("steps", default_steps))
            # Uvicorn may dispatch sync endpoints in parallel threads. Serialize model
            # access so multiple evaluation clients can safely share one GPU server.
            with inference_lock:
                with torch.no_grad():
                    action = model.generate_actions(**inputs, steps=steps).squeeze(0).float().cpu().numpy()
            return JSONResponse({"action": action.tolist()})
        except Exception as exc:
            logging.error(traceback.format_exc())
            return JSONResponse({"error": str(exc) or "Request failed"}, status_code=400)

    return app


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(args.dtype, device)

    processor_path = args.processor_path or args.model_path
    processor = VisionActionProcessor.from_pretrained(processor_path)
    model = BridgeWA.from_pretrained(args.model_path)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    if args.connection_info:
        _write_connection_info(
            Path(args.connection_info),
            host=args.advertise_host,
            port=int(args.port),
            model_path=args.model_path,
        )

    app = build_app(
        model,
        processor,
        device=device,
        dtype=dtype,
        default_steps=int(args.default_steps),
        default_domain_id=int(args.default_domain_id),
        image_layout=args.image_layout,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
