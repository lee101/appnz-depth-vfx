"""Depth Anything V2 Small as a reusable depth/VFX Cog."""

from pathlib import Path as LocalPath

from cog import BaseRunner, Input, Path

from contract import output_suffix, validate_dimensions


MODEL_DIR = "/weights/depth-anything-v2-small"
def normal_map(depth):
    import cv2
    import numpy as np

    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    vectors = np.dstack((-dx * 2.5, -dy * 2.5, np.ones_like(depth)))
    vectors /= np.maximum(np.linalg.norm(vectors, axis=2, keepdims=True), 1e-6)
    return ((vectors * 0.5 + 0.5) * 255).clip(0, 255).astype("uint8")


def write_parallax(image, depth, destination: LocalPath, strength: float, seconds: int) -> None:
    import cv2
    import numpy as np

    height, width = depth.shape
    fps = 24
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("could not initialize MP4 encoder")
    x, y = np.meshgrid(np.arange(width, dtype="float32"), np.arange(height, dtype="float32"))
    centered = depth.astype("float32") - float(depth.mean())
    amplitude = min(width, height) * 0.035 * strength
    try:
        for frame in range(fps * seconds):
            phase = 2 * np.pi * frame / (fps * seconds)
            offset_x = centered * amplitude * np.sin(phase)
            offset_y = centered * amplitude * 0.35 * np.cos(phase)
            rendered = cv2.remap(
                image,
                x + offset_x,
                y + offset_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            writer.write(rendered)
    finally:
        writer.release()


class Runner(BaseRunner):
    def setup(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForDepthEstimation.from_pretrained(
            MODEL_DIR, local_files_only=True, torch_dtype=dtype
        ).to(self.device).eval()
        if self.device.type == "cuda":
            self.model.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True

    def run(
        self,
        image: Path = Input(description="JPEG, PNG, or WebP image"),
        effect: str = Input(
            description="Depth-powered output",
            default="parallax",
            choices=["parallax", "depth", "heatmap", "normals"],
        ),
        strength: float = Input(description="Parallax displacement", default=0.55, ge=0.1, le=1.0),
        seconds: int = Input(description="Parallax duration", default=3, ge=1, le=6),
    ) -> Path:
        import cv2
        import numpy as np
        from PIL import Image

        suffix = output_suffix(effect)
        source = Image.open(str(image)).convert("RGB")
        validate_dimensions(*source.size)
        inputs = self.processor(images=source, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.device.type == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half().contiguous(memory_format=self.torch.channels_last)
        with self.torch.inference_mode():
            prediction = self.model(**inputs).predicted_depth
            prediction = self.torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=source.size[::-1], mode="bicubic", align_corners=False
            ).squeeze()
        raw = prediction.float().cpu().numpy()
        raw = (raw - raw.min()) / max(float(raw.max() - raw.min()), 1e-6)
        destination = LocalPath("/tmp") / f"{LocalPath(str(image)).stem}-{effect}{suffix}"
        if effect == "depth":
            cv2.imwrite(str(destination), (raw * 255).astype("uint8"))
        elif effect == "heatmap":
            cv2.imwrite(str(destination), cv2.applyColorMap((raw * 255).astype("uint8"), cv2.COLORMAP_TURBO))
        elif effect == "normals":
            cv2.imwrite(str(destination), cv2.cvtColor(normal_map(raw), cv2.COLOR_RGB2BGR))
        else:
            frame = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR)
            write_parallax(frame, raw, destination, strength, seconds)
        return Path(destination)
