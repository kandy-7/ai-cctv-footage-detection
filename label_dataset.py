
import csv
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "datasets" / "appearance" / "upper_color" / "incoming"
OUTPUT = ROOT / "datasets" / "appearance" / "upper_color" / "reviewed_labels.csv"

LABELS = {
    "1": "orange",
    "2": "pink",
    "3": "red",
    "4": "blue",
    "5": "black",
    "6": "white",
    "7": "other",
    "8": "uncertain"
}

# Find all extracted images
images = sorted(
    p for p in SOURCE.rglob("*")
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
)

# Load previously reviewed labels
reviewed = {}

if OUTPUT.exists():
    with open(OUTPUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reviewed[row["image_path"]] = row["label"]

# Save labels without modifying original images
def save_labels():
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "label"]
        )
        writer.writeheader()

        for path, label in reviewed.items():
            writer.writerow({
                "image_path": path,
                "label": label
            })


class LabelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CCTV Appearance Dataset Labeller")
        self.root.geometry("950x750")
        self.root.configure(bg="#171717")

        self.index = 0
        self.history = []

        self.title = tk.Label(
            root,
            text="CCTV Clothing Colour Labeller",
            font=("Arial", 20, "bold"),
            fg="white",
            bg="#171717"
        )
        self.title.pack(pady=12)

        self.image_label = tk.Label(
            root,
            bg="#252525",
            width=700,
            height=440
        )
        self.image_label.pack(pady=10)

        self.info = tk.Label(
            root,
            text="",
            font=("Arial", 11),
            fg="white",
            bg="#171717"
        )
        self.info.pack(pady=5)

        self.status = tk.Label(
            root,
            text="",
            font=("Arial", 12, "bold"),
            fg="#f0a43c",
            bg="#171717"
        )
        self.status.pack(pady=5)

        buttons = tk.Frame(root, bg="#171717")
        buttons.pack(pady=15)

        for key, label in LABELS.items():
            tk.Button(
                buttons,
                text=f"{key} - {label.upper()}",
                width=12,
                command=lambda value=label: self.assign(value)
            ).grid(
                row=int(key) // 4,
                column=int(key) % 4,
                padx=5,
                pady=5
            )

        tk.Button(
            root,
            text="UNDO LAST LABEL",
            command=self.undo,
            width=20
        ).pack(pady=5)

        self.root.bind("<Key>", self.handle_key)

        self.show_image()

    def show_image(self):
        # Skip images already labelled
        while self.index < len(images):
            path = str(images[self.index])

            if path not in reviewed:
                break

            self.index += 1

        if self.index >= len(images):
            self.image_label.config(image="", text="ALL IMAGES REVIEWED")
            self.info.config(text="Dataset review completed.")
            self.status.config(text=f"Saved labels: {len(reviewed)}")
            return

        path = images[self.index]

        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((700, 440))

            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo)

            relative = path.relative_to(SOURCE)

            self.info.config(
                text=f"{self.index + 1}/{len(images)} | {relative}"
            )

            self.status.config(
                text=f"Reviewed: {len(reviewed)} | "
                     f"Remaining: {len(images) - len(reviewed)}"
            )

        except Exception as error:
            print("Image error:", path, error)
            self.index += 1
            self.show_image()

    def assign(self, label):
        if self.index >= len(images):
            return

        path = str(images[self.index])

        reviewed[path] = label
        self.history.append(path)

        save_labels()

        print(f"{Path(path).name} -> {label}")

        self.index += 1
        self.show_image()

    def undo(self):
        if not self.history:
            return

        path = self.history.pop()

        reviewed.pop(path, None)
        save_labels()

        self.index = images.index(Path(path))
        self.show_image()

    def handle_key(self, event):
        key = event.char.lower()

        if key in LABELS:
            self.assign(LABELS[key])
        elif key == "u":
            self.undo()
        elif key == "s":
            self.index += 1
            self.show_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = LabelApp(root)
    root.mainloop()
