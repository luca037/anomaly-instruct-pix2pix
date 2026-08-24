"""
Compute the Kernel Inception Distance (KID) between real MVTec anomaly
images and generated defect images.

Adapted from:
    https://github.com/echrao/O2MAG/blob/main/eval/compute-kid.py

Differences vs. the upstream script:
  * Only the specified --categories are evaluated (default: the 7 classes
    [cable, screw, transistor, leather, hazelnut, pill, tile]).
  * Both real and generated images are expected to follow the *MVTec*
    folder layout:  <root>/<category>/test/<defect_class>/<images...>
  * Uses the modern torchvision `weights=` API for InceptionV3.
  * Drops the unused scipy import (the polynomial-kernel KID is pure numpy).
  * Pins the model to ``cuda:1`` only (per repo convention; never cuda:0).

Usage
-----
    # Real images live at /home/luca_piai/big_disk/datasets/mvtec
    # Generated images mirror the same <category>/test/<defect_class> layout
    uv run eval/compute_kid.py \
        --real_path /home/luca_piai/big_disk/datasets/mvtec \
        --generated_path /path/to/generated \
        --categories cable screw transistor leather hazelnut pill tile
"""

import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3

parser = argparse.ArgumentParser()
parser.add_argument(
    "--real_path",
    type=str,
    default="/home/luca_piai/big_disk/datasets/mvtec",
    help="Path to the real MVTec dataset (default: /home/luca_piai/.../mvtec)",
)
parser.add_argument(
    "--generated_path",
    type=str,
    default="/home/luca_piai/big_disk/datasets/generated",
    help="Path to the generated image dataset (mirrors MVTec <cat>/test/<class>)",
)
parser.add_argument(
    "--categories",
    type=str,
    nargs="+",
    default=["cable", "screw", "transistor", "leather", "hazelnut", "pill", "tile"],
    help="MVTec categories to evaluate",
)
parser.add_argument(
    "--subsample_size", type=int, default=10, help="Number of images per KID subset"
)
parser.add_argument(
    "--num_subsets",
    type=int,
    default=50,
    help="Number of KID subsets averaged per class",
)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:1",
    help="Device for computation (e.g. cuda:1).",
)
args = parser.parse_args()

logging.basicConfig(
    filename=f"{args.generated_path}_kid_score_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")


class InceptionV3FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False
        ).to(device)
        self.model.eval()

    def forward(self, x):
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def get_image_paths_from_folder(folder):
    image_paths = []
    if not os.path.isdir(folder):
        return image_paths
    for file in sorted(os.listdir(folder)):
        if file.lower().endswith(IMAGE_EXTS) and os.path.isfile(
            os.path.join(folder, file)
        ):
            image_paths.append(os.path.join(folder, file))
    return image_paths


def get_activations(image_paths, model, batch_size=32):
    activations = []
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(transform(img))
                except Exception as e:
                    print(f"Image loading error {path}: {e}")

            if batch_images:
                batch_images = torch.stack(batch_images).to(device)
                batch_activations = model(batch_images)
                activations.append(batch_activations.cpu().numpy())

    return np.concatenate(activations, axis=0) if activations else np.array([])


def polynomial_kernel_matrix(X, Y):
    """Vectorized polynomial kernel K(x,y) = (1 + <x,y>/d)^3, where d = feat dim."""
    d = X.shape[1]
    return (1.0 + np.dot(X, Y.T) / d) ** 3


def mmd_squared(x_feats, y_feats):
    """Unbiased polynomial-kernel MMD^2 (equivalent to the KID statistic)."""
    n, m = len(x_feats), len(y_feats)
    if n < 2 or m < 2:
        return None

    K_xx = polynomial_kernel_matrix(x_feats, x_feats)
    K_yy = polynomial_kernel_matrix(y_feats, y_feats)
    K_xy = polynomial_kernel_matrix(x_feats, y_feats)

    mmd = (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1))
    mmd += (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1))
    mmd -= 2.0 * K_xy.mean()
    return mmd


def calculate_kid(
    activations_real, activations_generated, subsample_size=None, num_subsets=50
):
    if activations_real.size == 0 or activations_generated.size == 0:
        return None
    if subsample_size is None:
        subsample_size = min(10, activations_real.shape[0])

    n_real = activations_real.shape[0]
    n_gen = activations_generated.shape[0]

    kid_scores = []
    for _ in range(num_subsets):
        idx_real = np.random.choice(
            n_real, min(subsample_size, n_real), replace=n_real < subsample_size
        )
        idx_gen = np.random.choice(
            n_gen, min(subsample_size, n_gen), replace=n_gen < subsample_size
        )
        mmd_val = mmd_squared(
            activations_real[idx_real], activations_generated[idx_gen]
        )
        if mmd_val is not None:
            kid_scores.append(mmd_val)

    return float(np.mean(kid_scores)) if kid_scores else None


def discover_defect_classes(generated_category_path):
    """Defect classes live directly under the category's ``test/`` folder
    (mirrors the MVTec layout: <category>/test/<defect_class>)."""
    test_folder = os.path.join(generated_category_path, "test")
    if not os.path.isdir(test_folder):
        # Tolerate a flat <category>/<defect_class> layout as a fallback.
        test_folder = generated_category_path
    if not os.path.isdir(test_folder):
        return []
    return [
        d
        for d in sorted(os.listdir(test_folder))
        if os.path.isdir(os.path.join(test_folder, d))
    ]


def main():
    real_root = args.real_path
    generated_root = args.generated_path
    model = InceptionV3FeatureExtractor().to(device)
    category_kid_scores = {}

    for category in args.categories:
        logging.info(f"Processing category: {category}")
        print(f"Processing category: {category}")
        class_kid_scores = []

        generated_category_path = os.path.join(generated_root, category)
        defect_classes = discover_defect_classes(generated_category_path)

        for defect_class in defect_classes:
            real_class_folder = os.path.join(real_root, category, "test", defect_class)
            generated_class_folder = os.path.join(
                generated_category_path, "test", defect_class
            )
            if not os.path.isdir(generated_class_folder):
                generated_class_folder = os.path.join(
                    generated_category_path, defect_class
                )

            real_image_paths = get_image_paths_from_folder(real_class_folder)
            generated_image_paths = get_image_paths_from_folder(generated_class_folder)

            if not real_image_paths:
                logging.warning(f"No real images for {category}/{defect_class}.")
                print(f"No real images for {category}/{defect_class}.")
                continue
            if not generated_image_paths:
                logging.warning(f"No generated images for {category}/{defect_class}.")
                print(f"No generated images for {category}/{defect_class}.")
                continue

            act_real = get_activations(
                real_image_paths, model, batch_size=args.batch_size
            )
            act_generated = get_activations(
                generated_image_paths, model, batch_size=args.batch_size
            )

            if act_real.size == 0 or act_generated.size == 0:
                logging.warning(f"No valid activations for {category}/{defect_class}.")
                print(f"No valid activations for {category}/{defect_class}.")
                continue

            subsample_size = min(args.subsample_size, act_real.shape[0])
            kid_score = calculate_kid(
                act_real,
                act_generated,
                subsample_size=subsample_size,
                num_subsets=args.num_subsets,
            )
            if kid_score is not None:
                class_kid_scores.append(kid_score)
                logging.info(
                    f"KID score for {category}/{defect_class}: {kid_score * 1000}"
                )
                print(f"KID score for {category}/{defect_class}: {kid_score * 1000}")

        if class_kid_scores:
            category_mean_kid = float(np.mean(class_kid_scores))
            category_kid_scores[category] = category_mean_kid * 1000
            print(f"Mean KID score for {category}: {category_mean_kid * 1000}")
            logging.info(f"Mean KID score for {category}: {category_mean_kid * 1000}")
        else:
            logging.warning(f"No valid images for category {category}.")
            print(f"No valid images for category {category}.")

    print("\nMean KID score per category:")
    logging.info("Mean KID score per category:")
    for category, kid in category_kid_scores.items():
        print(f"{category}: {kid}")
        logging.info(f"{category}: {kid}")

    if category_kid_scores:
        overall_mean_kid = float(np.mean(list(category_kid_scores.values())))
        print(f"\nOverall mean KID score: {overall_mean_kid}")
        logging.info(f"Overall mean KID score: {overall_mean_kid}")


if __name__ == "__main__":
    main()
