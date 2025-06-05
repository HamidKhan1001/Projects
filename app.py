# app.py
import os
from io import BytesIO
from PIL import Image

import torch
from flask import Flask, request, render_template, redirect, url_for, flash

from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)
app.secret_key = "556"

# ─── 3.1. Load the BLIP image‐captioning model at startup ───────────────────
MODEL_NAME = "Salesforce/blip-image-captioning-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model once (so you don't reload on every request)
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)


def generate_caption(pil_image: Image.Image) -> str:
    """
    Given a PIL image, run it through the BLIP model and return the caption string.
    """
    # Preprocess: convert PIL → pixel values tensor
    inputs = processor(pil_image, return_tensors="pt").to(device)
    # Generate token IDs with the model
    out = model.generate(**inputs)
    # Decode the first generated sequence into text
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# ─── 3.2. Routes ─────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    """
    GET:  Render an upload form.
    POST: Accept uploaded image, generate caption, and re‐render with caption.
    """
    caption = None

    if request.method == "POST":
        # 1. Ensure a file was submitted
        if "image_file" not in request.files:
            flash("No file part in request.", "error")
            return redirect(request.url)

        file = request.files["image_file"]
        # 2. If user submitted an empty file
        if file.filename == "":
            flash("No file selected.", "error")
            return redirect(request.url)

        # 3. Verify it’s an image
        if not file.content_type.startswith("image/"):
            flash("Please upload a valid image (PNG/JPG).", "error")
            return redirect(request.url)

        # 4. Read file into PIL, generate caption
        try:
            # Read file bytes → PIL Image
            img_bytes = file.read()
            pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")

            # Optionally: resize large images to speed up inference
            max_dim = 512
            if min(pil_img.size) > max_dim:
                pil_img.thumbnail((max_dim, max_dim))

            caption = generate_caption(pil_img)
        except Exception as e:
            flash(f"Error during caption generation: {e}", "error")
            return redirect(request.url)

    # Render index.html, passing caption (None if GET)
    return render_template("index.html", caption=caption)


if __name__ == "__main__":
    # By default, Flask runs on port 5000 in debug mode
    app.run(host="0.0.0.0", port=5000, debug=True)
