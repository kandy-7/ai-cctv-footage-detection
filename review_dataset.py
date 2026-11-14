
from pathlib import Path
from PIL import Image, ImageDraw
import math

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "datasets/appearance/upper_color/incoming"
OUTPUT = ROOT / "datasets/appearance/contact_sheet.jpg"

files = []

for folder in SOURCE.iterdir():
    if folder.is_dir():
        for image in folder.glob("*.jpg"):
            files.append((image, folder.name))

thumb_w, thumb_h = 120, 150
label_h = 25
columns = 6
rows = math.ceil(len(files) / columns)

sheet = Image.new(
    "RGB",
    (columns * thumb_w, rows * (thumb_h + label_h)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, (path, label) in enumerate(files):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w - 8, thumb_h - 8))

        x = (i % columns) * thumb_w
        y = (i // columns) * (thumb_h + label_h)

        sheet.paste(img, (x + 4, y + 4))
        draw.text(
            (x + 4, y + thumb_h),
            label,
            fill="black"
        )
    except Exception as e:
        print("Skipped:", path, e)

sheet.save(OUTPUT, quality=90)

print("Images:", len(files))
print("Contact sheet:", OUTPUT)
