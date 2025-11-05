import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from typing import Tuple, List, Optional
import argparse
from tqdm import tqdm
import json


class LowLightDatasetPreprocessor:
    def __init__(self, output_dir: str = "processed_dataset", target_size: Tuple[int, int] = (512, 512)):
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.train_dir = self.output_dir / "train"
        self.test_dir = self.output_dir / "test"
        self.visualizations_dir = self.output_dir / "visualizations"
        self._create_directories()
    
    def _create_directories(self):
        for dir_path in [self.train_dir, self.test_dir, self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def resize_image(self, image: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        return cv2.resize(image, self.target_size, interpolation=interpolation)
    
    def normalize_image(self, image: np.ndarray, method: str = "standard") -> np.ndarray:
        if method == "standard":
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            std = np.std(image, axis=(0, 1), keepdims=True)
            std = np.where(std == 0, 1, std)
            normalized = (image - mean) / std
            normalized = ((normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8) * 255).astype(np.uint8)
        elif method == "minmax":
            min_val = image.min()
            max_val = image.max()
            if max_val - min_val > 0:
                normalized = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized = image
        else:
            normalized = (image.astype(np.float32) / 255.0)
            normalized = (normalized * 255).astype(np.uint8)
        return normalized
    
    def synthetic_darkening(self, image: np.ndarray, darkening_factor: float = 0.3, 
                           gamma: float = 2.0, noise_level: float = 0.05) -> np.ndarray:
        img_float = image.astype(np.float32) / 255.0
        darkened = img_float * darkening_factor
        darkened = np.power(darkened, gamma)
        noise = np.random.normal(0, noise_level, darkened.shape)
        darkened = np.clip(darkened + noise, 0, 1)
        return (darkened * 255).astype(np.uint8)
    
    def augment_image(self, image: np.ndarray, apply_darkening: bool = True) -> List[np.ndarray]:
        augmented = [image]
        if apply_darkening:
            for factor in [0.2, 0.3, 0.4]:
                for gamma in [1.8, 2.0, 2.2]:
                    darkened = self.synthetic_darkening(image, darkening_factor=factor, gamma=gamma)
                    augmented.append(darkened)
        return augmented
    
    def process_image(self, image_path: Path, normalize: bool = True, 
                     augment: bool = False) -> List[Tuple[np.ndarray, str]]:
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        resized = self.resize_image(image)
        processed = self.normalize_image(resized, method="standard") if normalize else resized
        
        results = []
        base_name = image_path.stem
        
        if augment:
            augmented_images = self.augment_image(processed, apply_darkening=True)
            for idx, aug_img in enumerate(augmented_images):
                filename = f"{base_name}.jpg" if idx == 0 else f"{base_name}_aug{idx}.jpg"
                results.append((aug_img, filename))
        else:
            results.append((processed, f"{base_name}.jpg"))
        return results
    
    def process_dataset(self, input_dir: str, split_ratio: float = 0.8, 
                       normalize: bool = True, augment: bool = True,
                       test_size: Optional[int] = None) -> dict:
        input_path = Path(input_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'**/*{ext}'))
            image_files.extend(input_path.glob(f'**/*{ext.upper()}'))
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {input_dir}")
        
        print(f"Found {len(image_files)} images")
        
        if test_size:
            test_files = list(image_files[:test_size])
            train_files = list(image_files[test_size:])
        else:
            train_files, test_files = train_test_split(
                image_files, test_size=1-split_ratio, random_state=42
            )
        
        print(f"Train: {len(train_files)} images, Test: {len(test_files)} images")
        
        train_count = 0
        print("\nProcessing training set...")
        for img_path in tqdm(train_files, desc="Train"):
            processed = self.process_image(img_path, normalize=normalize, augment=augment)
            for img, filename in processed:
                cv2.imwrite(str(self.train_dir / filename), img)
                train_count += 1
        
        test_count = 0
        print("\nProcessing test set...")
        for img_path in tqdm(test_files, desc="Test"):
            processed = self.process_image(img_path, normalize=normalize, augment=False)
            for img, filename in processed:
                cv2.imwrite(str(self.test_dir / filename), img)
                test_count += 1
        
        stats = {
            "original_images": len(image_files),
            "train_images": len(train_files),
            "test_images": len(test_files),
            "processed_train": train_count,
            "processed_test": test_count,
            "augmentation_applied": augment
        }
        
        with open(self.output_dir / "dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nProcessing done!")
        print(f"Train: {train_count} processed images")
        print(f"Test: {test_count} processed images")
        return stats
    
    def create_visual_examples(self, num_examples: int = 5, source: str = "train"):
        source_dir = self.train_dir if source == "train" else self.test_dir
        image_files = list(source_dir.glob("*.jpg"))[:num_examples]
        
        if len(image_files) == 0:
            return
        
        fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, img_path in enumerate(image_files):
            processed = cv2.imread(str(img_path))
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            axes[idx, 0].imshow(processed_rgb)
            axes[idx, 0].set_title(f"Processed\n{img_path.name}")
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(processed_rgb)
            axes[idx, 1].set_title(f"Resized {self.target_size}")
            axes[idx, 1].axis('off')
            
            normalized = self.normalize_image(processed, method="standard")
            normalized_rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
            axes[idx, 2].imshow(normalized_rgb)
            axes[idx, 2].set_title("Normalized")
            axes[idx, 2].axis('off')
            
            darkened = self.synthetic_darkening(processed, darkening_factor=0.3, gamma=2.0)
            darkened_rgb = cv2.cvtColor(darkened, cv2.COLOR_BGR2RGB)
            axes[idx, 3].imshow(darkened_rgb)
            axes[idx, 3].set_title("Synthetic Darkening")
            axes[idx, 3].axis('off')
        
        plt.tight_layout()
        output_path = self.visualizations_dir / f"preprocessing_examples_{source}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visual examples saved to {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Low-Light Image Dataset Preprocessing")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="processed_dataset")
    parser.add_argument("--target_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--num_examples", type=int, default=5)
    
    args = parser.parse_args()
    
    preprocessor = LowLightDatasetPreprocessor(
        output_dir=args.output_dir,
        target_size=tuple(args.target_size)
    )
    
    stats = preprocessor.process_dataset(
        input_dir=args.input_dir,
        split_ratio=args.split_ratio,
        normalize=not args.no_normalize,
        augment=not args.no_augment,
        test_size=args.test_size
    )
    
    preprocessor.create_visual_examples(num_examples=args.num_examples, source="train")
    preprocessor.create_visual_examples(num_examples=args.num_examples, source="test")
    
    print("\n" + "="*50)
    print("Dataset preprocessing done!")
    print("="*50)
    print(f"Output directory: {args.output_dir}")
    print(f"Train images: {stats['processed_train']}")
    print(f"Test images: {stats['processed_test']}")
    print(f"Visualizations: {preprocessor.visualizations_dir}")


if __name__ == "__main__":
    main()

