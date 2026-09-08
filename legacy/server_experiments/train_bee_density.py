"""用蜜蜂关键点/框生成密度监督，训练轻量全图密度与计数模型。"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def gaussian_kernel(sigma=1.5):
    radius = int(math.ceil(3 * sigma))
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    kernel = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    return kernel / kernel.sum(), radius


class BeeDensityDataset(Dataset):
    def __init__(self, annotations, image_root, size, augment=False):
        data = json.loads(Path(annotations).read_text(encoding="utf-8"))
        self.images = sorted(data["images"], key=lambda item: item["id"])
        self.by_image = defaultdict(list)
        for annotation in data["annotations"]:
            self.by_image[annotation["image_id"]].append(annotation)
        self.image_root = Path(image_root)
        self.width, self.height = size
        self.out_width, self.out_height = self.width // 4, self.height // 4
        self.augment = augment
        self.kernel, self.radius = gaussian_kernel()

    def __len__(self):
        return len(self.images)

    def density_map(self, image, annotations):
        density = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        scale_x = self.out_width / image["width"]
        scale_y = self.out_height / image["height"]
        for annotation in annotations:
            keypoints = annotation.get("keypoints", [])
            if len(keypoints) >= 6 and keypoints[2] > 0 and keypoints[5] > 0:
                x = (keypoints[0] + keypoints[3]) * 0.5
                y = (keypoints[1] + keypoints[4]) * 0.5
            else:
                box = annotation["bbox"]
                x, y = box[0] + box[2] * 0.5, box[1] + box[3] * 0.5
            cx = int(round(x * scale_x))
            cy = int(round(y * scale_y))
            x1, x2 = max(0, cx - self.radius), min(self.out_width, cx + self.radius + 1)
            y1, y2 = max(0, cy - self.radius), min(self.out_height, cy + self.radius + 1)
            if x1 >= x2 or y1 >= y2:
                continue
            kx1, ky1 = x1 - (cx - self.radius), y1 - (cy - self.radius)
            kx2, ky2 = kx1 + (x2 - x1), ky1 + (y2 - y1)
            patch = self.kernel[ky1:ky2, kx1:kx2]
            density[y1:y2, x1:x2] += patch / max(float(patch.sum()), 1e-6)
        return density

    def __getitem__(self, index):
        record = self.images[index]
        image_path = Path(record["file_name"])
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        image = Image.open(image_path).convert("RGB").resize((self.width, self.height))
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        density = self.density_map(record, self.by_image[record["id"]])

        if self.augment and random.random() < 0.5:
            array = array[:, ::-1].copy()
            density = density[:, ::-1].copy()
        if self.augment:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.08, 0.08)
            array = np.clip(array * gain + bias, 0, 1)

        tensor = torch.from_numpy(array.transpose(2, 0, 1))
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return (tensor - mean) / std, torch.from_numpy(density[None]), float(len(self.by_image[record["id"]]))


class DensityNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.lateral = nn.ModuleList([
            nn.Conv2d(64, 64, 1), nn.Conv2d(128, 64, 1),
            nn.Conv2d(256, 64, 1), nn.Conv2d(512, 64, 1),
        ])
        final_layer = nn.Conv2d(64, 1, 1)
        nn.init.constant_(final_layer.bias, -5.0)
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            final_layer, nn.Softplus(),
        )

    def forward(self, image):
        x1 = self.layer1(self.stem(image))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        target_size = x1.shape[-2:]
        features = self.lateral[0](x1)
        for layer, feature in zip(self.lateral[1:], (x2, x3, x4)):
            features = features + F.interpolate(layer(feature), target_size, mode="bilinear", align_corners=False)
        return self.head(features)


def loss_function(prediction, target, count):
    density_loss = F.mse_loss(prediction, target) * 1000.0
    predicted_count = prediction.sum(dim=(1, 2, 3))
    count_loss = F.smooth_l1_loss(predicted_count / 300.0, count / 300.0)
    return density_loss + count_loss, density_loss, count_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    errors = []
    squared = []
    for image, _, count in loader:
        prediction = model(image.to(device, non_blocking=True))
        predicted_count = prediction.sum(dim=(1, 2, 3)).cpu()
        error = predicted_count - count
        errors.extend(error.abs().tolist())
        squared.extend(error.square().tolist())
    return float(np.mean(errors)), float(math.sqrt(np.mean(squared)))


@torch.no_grad()
def save_density_example(model, dataset, device, output_path):
    model.eval()
    image, target, count = dataset[0]
    prediction = model(image[None].to(device)).squeeze().cpu().numpy()
    target = target.squeeze().numpy()
    mean = np.asarray([0.485, 0.456, 0.406])[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225])[:, None, None]
    visible_image = np.clip(image.numpy() * std + mean, 0, 1).transpose(1, 2, 0)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(visible_image)
    axes[0].set_title(f"Input (GT count={count:.0f})")
    axes[1].imshow(target, cmap="magma")
    axes[1].set_title("Ground-truth density")
    axes[2].imshow(prediction, cmap="magma")
    axes[2].set_title(f"Predicted density (count={prediction.sum():.1f})")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    size = (args.width, args.height)
    train_set = BeeDensityDataset(args.train_annotations, args.image_root, size, augment=True)
    val_set = BeeDensityDataset(args.val_annotations, args.image_root, size, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DensityNet(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_mae = float("inf")
    log_path = args.output_dir / "metrics.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = []
        for image, target, count in train_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            count = count.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(image)
                loss, density_loss, count_loss = loss_function(prediction, target, count)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals.append((float(loss), float(density_loss), float(count_loss)))
        scheduler.step()
        mae, rmse = evaluate(model, val_loader, device)
        values = np.asarray(totals).mean(axis=0)
        record = {
            "epoch": epoch, "loss": float(values[0]),
            "density_loss": float(values[1]), "count_loss": float(values[2]),
            "val_mae": mae, "val_rmse": rmse, "lr": scheduler.get_last_lr()[0],
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        checkpoint = {"model": model.state_dict(), "args": vars(args), "metrics": record}
        torch.save(checkpoint, args.output_dir / "latest.pth")
        if mae < best_mae:
            best_mae = mae
            torch.save(checkpoint, args.output_dir / "best_mae.pth")

    best = torch.load(args.output_dir / "best_mae.pth", map_location=device)
    model.load_state_dict(best["model"])
    save_density_example(model, val_set, device, args.output_dir / "density_example.png")


if __name__ == "__main__":
    main()
