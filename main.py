import os
import csv
import torch
import numpy as np
import onnxruntime as ort
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from torch.hub import download_url_to_file

# Import the BLIP model structure (ensure you have the necessary BLIP files or install the BLIP package)
from extras.BLIP.models.blip import blip_decoder

# Disable warnings from onnxruntime
ort.set_default_logger_severity(3)

# Configuration for BLIP
blip_image_eval_size = 384
blip_model_url = 'https://huggingface.co/lllyasviel/misc/resolve/main/model_base_caption_capfilt_large.pth'
blip_model_filename = 'model_base_caption_capfilt_large.pth'

# Configuration for Anime Tagging
anime_model_url = 'https://huggingface.co/lllyasviel/misc/resolve/main/wd-v1-4-moat-tagger-v2.onnx'
anime_model_csv_url = 'https://huggingface.co/lllyasviel/misc/resolve/main/wd-v1-4-moat-tagger-v2.csv'
anime_model_filename = 'wd-v1-4-moat-tagger-v2.onnx'
anime_model_csv_filename = 'wd-v1-4-moat-tagger-v2.csv'

# Ensure the model directory exists
model_dir = 'models'


def load_blip_model():
    blip_model_path = os.path.join(model_dir, blip_model_filename)

    if not os.path.exists(blip_model_path):
        download_url_to_file(blip_model_url, blip_model_path)

    blip_model = blip_decoder(pretrained=blip_model_path, image_size=blip_image_eval_size, vit='base')
    blip_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blip_model = blip_model.to(device)
    return blip_model, device


def generate_photo_caption(blip_model, device, img_rgb):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        caption = blip_model.generate(img_tensor, sample=True, num_beams=1, max_length=75, min_length=5)[0]
    return caption


def load_anime_model_and_tags():
    anime_model_path = os.path.join(model_dir, anime_model_filename)
    anime_csv_path = os.path.join(model_dir, anime_model_csv_filename)

    if not os.path.exists(anime_model_path):
        download_url_to_file(anime_model_url, anime_model_path)
    if not os.path.exists(anime_csv_path):
        download_url_to_file(anime_model_csv_url, anime_csv_path)

    anime_model = ort.InferenceSession(anime_model_path, providers=ort.get_available_providers())

    anime_tags = []
    with open(anime_csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            anime_tags.append(row[1])
    return anime_model, anime_tags


def preprocess_for_anime_model(img_rgb, target_size=448):
    img = img_rgb.resize((target_size, target_size), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32)
    img_np = img_np / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)
    img_np = np.transpose(img_np, (0, 2, 3, 1))
    return img_np


def get_tags_from_anime_model(anime_model, img_np):
    input_name = anime_model.get_inputs()[0].name
    output_name = anime_model.get_outputs()[0].name
    preds = anime_model.run([output_name], {input_name: img_np})[0]
    return preds


def generate_anime_caption(anime_model, anime_tags, img_rgb):
    img_np = preprocess_for_anime_model(img_rgb)
    preds = get_tags_from_anime_model(anime_model, img_np)[0]  # Assuming single output
    threshold = 0.5
    filtered_tags = [tag for tag, score in zip(anime_tags, preds) if score > threshold]
    caption = ', '.join(filtered_tags)
    return caption


def load_image(image_path):
    return Image.open(image_path).convert('RGB')


if __name__ == "__main__":
    image_path = "images/image.png"
    img_rgb = load_image(image_path)

    blip_model, device = load_blip_model()
    caption = generate_photo_caption(blip_model, device, img_rgb)
    print("Photo Caption:", caption)

    anime_model, anime_tags = load_anime_model_and_tags()
    caption = generate_anime_caption(anime_model, anime_tags, img_rgb)
    print("Anime Caption:", caption)
