from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from src.config import ImageConfig


@dataclass(frozen=True)
class ImageTransform:
    resize_mode: str
    target_long_edge: int
    pad_fill: tuple[int, int, int] = (0, 0, 0)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.resize_mode == "stretch":
            return image.resize((self.target_long_edge, self.target_long_edge), Image.BICUBIC)
        if self.resize_mode == "aspect_pad":
            return self._aspect_pad(image)
        raise ValueError(f"Unsupported resize mode: {self.resize_mode}")

    def _aspect_pad(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.target_long_edge / max(width, height)
        resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BICUBIC)
        canvas = Image.new("RGB", (self.target_long_edge, self.target_long_edge), self.pad_fill)
        offset_x = (self.target_long_edge - resized.width) // 2
        offset_y = (self.target_long_edge - resized.height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas


def build_image_transform(config: ImageConfig) -> ImageTransform:
    return ImageTransform(
        resize_mode=config.resize_mode,
        target_long_edge=config.target_long_edge,
    )

