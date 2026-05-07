from __future__ import annotations

from dataclasses import dataclass
import random

from PIL import Image, ImageEnhance

from src.config import ImageConfig


@dataclass(frozen=True)
class ImageTransform:
    resize_mode: str
    target_long_edge: int
    augment: bool = False
    random_resized_crop_scale: tuple[float, float] = (0.9, 1.0)
    rotation_degrees: float = 3.0
    brightness_range: tuple[float, float] = (0.9, 1.1)
    pad_fill: tuple[int, int, int] = (0, 0, 0)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.resize_mode == "stretch":
            transformed = image.resize((self.target_long_edge, self.target_long_edge), Image.BICUBIC)
        elif self.resize_mode == "aspect_pad":
            transformed = self._aspect_pad(image)
        else:
            raise ValueError(f"Unsupported resize mode: {self.resize_mode}")
        if self.augment:
            transformed = self._augment(transformed)
        return transformed

    def _aspect_pad(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.target_long_edge / max(width, height)
        resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BICUBIC)
        canvas = Image.new("RGB", (self.target_long_edge, self.target_long_edge), self.pad_fill)
        offset_x = (self.target_long_edge - resized.width) // 2
        offset_y = (self.target_long_edge - resized.height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas

    def _augment(self, image: Image.Image) -> Image.Image:
        image = self._random_resized_crop(image)
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=self.pad_fill)
        brightness_min, brightness_max = self.brightness_range
        if brightness_min != 1.0 or brightness_max != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(brightness_min, brightness_max))
        return image

    def _random_resized_crop(self, image: Image.Image) -> Image.Image:
        scale_min, scale_max = self.random_resized_crop_scale
        crop_scale = random.uniform(scale_min, scale_max)
        if crop_scale >= 1.0:
            return image
        width, height = image.size
        crop_width = max(1, round(width * crop_scale))
        crop_height = max(1, round(height * crop_scale))
        left = random.randint(0, max(0, width - crop_width))
        top = random.randint(0, max(0, height - crop_height))
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), Image.BICUBIC)


def build_image_transform(config: ImageConfig, augment: bool = False) -> ImageTransform:
    return ImageTransform(
        resize_mode=config.resize_mode,
        target_long_edge=config.target_long_edge,
        augment=augment and config.augmentation.enabled,
        random_resized_crop_scale=tuple(config.augmentation.random_resized_crop_scale),
        rotation_degrees=config.augmentation.rotation_degrees,
        brightness_range=tuple(config.augmentation.brightness_range),
    )
