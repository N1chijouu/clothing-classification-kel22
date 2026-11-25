from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Konfigurasi yang digunakan bersama oleh training dan aplikasi.
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
SEED = 42
IMAGE_SIZE = 160
CLASSES = ("T-Shirt", "Longsleeve", "Pants", "Shirt", "Shoes", "Dress", "Shorts", "Outwear")

# Normalisasi sesuai bobot awal MobileNetV3 yang dilatih pada ImageNet.
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def read_image(source):
    """Baca foto, benarkan orientasi EXIF, dan ubah ke RGB."""
    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source).convert("RGB")
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def preprocess(training=False):
    """Resize baseline 160x160; augmentasi hanya saat belajar, bukan evaluasi."""
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]
    return transforms.Compose(steps + [transforms.ToTensor(), NORMALIZE])


def create_mobilenet(pretrained=False):
    """Encoder pretrained dipakai untuk fitur model klasik dan inisialisasi CNN."""
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    return mobilenet_v3_small(weights=weights)


def create_cnn(num_classes):
    """Ganti classifier terakhir agar menghasilkan skor untuk 8 kelas."""
    cnn = create_mobilenet(pretrained=False)
    cnn.classifier[3] = nn.Linear(cnn.classifier[3].in_features, num_classes)
    return cnn


@torch.inference_mode()
def extract_features(encoder, batch):
    """Ubah citra menjadi vektor fitur (576 dimensi) dari MobileNetV3."""
    return torch.flatten(encoder.avgpool(encoder.features(batch)), 1)


class ClothingDataset(Dataset):
    """Jembatan data CSV/path foto ke tensor yang dipakai PyTorch."""

    def __init__(self, frame, training=False):
        self.paths = list(frame["file_path"].astype(str))
        self.labels = [CLASSES.index(label) for label in frame["label"]]
        self.transform = preprocess(training=training)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = read_image(self.paths[index])
        return self.transform(image), self.labels[index]
