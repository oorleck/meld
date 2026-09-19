"""Draw the app icon (a sound wave in a rounded square) and save it as meld.ico."""
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent


def draw(size: int = 256) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, size - 1, size - 1), radius=size // 5, fill=(24, 26, 38, 255))
    bars = [0.25, 0.55, 0.85, 0.5, 1.0, 0.6, 0.8, 0.4, 0.2]  # heights of the wave
    n = len(bars)
    margin = size * 0.16
    slot = (size - 2 * margin) / n
    for i, h in enumerate(bars):
        x = margin + i * slot + slot * 0.18
        bar_h = (size * 0.62) * h
        y0 = size / 2 - bar_h / 2
        color = (255, 210, 63, 255) if i % 2 == 0 else (255, 120, 90, 255)
        d.rounded_rectangle((x, y0, x + slot * 0.64, y0 + bar_h), radius=slot * 0.3, fill=color)
    return img


if __name__ == "__main__":
    draw().save(HERE / "meld.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote", HERE / "meld.ico")
