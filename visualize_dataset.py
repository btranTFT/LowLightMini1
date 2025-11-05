import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import Counter


def plot_dataset_statistics(dataset_dir: Path, output_path: Path = None):
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    
    train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    def get_image_stats(img_dir):
        stats = {
            "mean_brightness": [],
            "std_brightness": [],
            "mean_r": [], "mean_g": [], "mean_b": [],
            "sizes": []
        }
        
        for img_path in img_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                stats["mean_brightness"].append(gray.mean())
                stats["std_brightness"].append(gray.std())
                stats["mean_r"].append(img[:, :, 2].mean())
                stats["mean_g"].append(img[:, :, 1].mean())
                stats["mean_b"].append(img[:, :, 0].mean())
                stats["sizes"].append(img.shape[:2])
        
        return stats
    
    train_stats = get_image_stats(train_dir)
    test_stats = get_image_stats(test_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].hist(train_stats["mean_brightness"], bins=50, alpha=0.7, label="Train", color="blue")
    axes[0, 0].hist(test_stats["mean_brightness"], bins=50, alpha=0.7, label="Test", color="red")
    axes[0, 0].set_xlabel("Mean Brightness")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Brightness Distribution")
    axes[0, 0].legend()
    
    axes[0, 1].boxplot([train_stats["mean_r"], train_stats["mean_g"], train_stats["mean_b"]], 
                       tick_labels=["R", "G", "B"])
    axes[0, 1].set_ylabel("Mean Pixel Value")
    axes[0, 1].set_title("RGB Channel Statistics (Train)")
    
    train_sizes = Counter(train_stats["sizes"])
    size_labels = [f"{h}x{w}" for (h, w), _ in train_sizes.most_common(5)]
    size_counts = [count for _, count in train_sizes.most_common(5)]
    axes[0, 2].bar(range(len(size_labels)), size_counts)
    axes[0, 2].set_xticks(range(len(size_labels)))
    axes[0, 2].set_xticklabels(size_labels, rotation=45)
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].set_title("Image Size Distribution")
    
    axes[1, 0].pie([len(train_images), len(test_images)], 
                   labels=["Train", "Test"], 
                   autopct='%1.1f%%',
                   colors=["blue", "red"])
    axes[1, 0].set_title("Train/Test Split")
    
    axes[1, 1].scatter(train_stats["mean_brightness"], train_stats["std_brightness"], 
                      alpha=0.5, s=10, label="Train")
    axes[1, 1].scatter(test_stats["mean_brightness"], test_stats["std_brightness"], 
                      alpha=0.5, s=10, label="Test", color="red")
    axes[1, 1].set_xlabel("Mean Brightness")
    axes[1, 1].set_ylabel("Std Brightness")
    axes[1, 1].set_title("Brightness vs Variability")
    axes[1, 1].legend()
    
    axes[1, 2].axis('off')
    summary_text = f"""
    Dataset Statistics Summary:
    
    Total Images: {len(train_images) + len(test_images)}
    Train Images: {len(train_images)}
    Test Images: {len(test_images)}
    
    Train Stats:
    Mean Brightness: {np.mean(train_stats['mean_brightness']):.2f}
    Std Brightness: {np.std(train_stats['mean_brightness']):.2f}
    
    Test Stats:
    Mean Brightness: {np.mean(test_stats['mean_brightness']):.2f}
    Std Brightness: {np.std(test_stats['mean_brightness']):.2f}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, 
                    verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Statistics plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_sample_grid(dataset_dir: Path, num_samples: int = 8, output_path: Path = None):
    train_dir = dataset_dir / "train"
    images = list(train_dir.glob("*.jpg"))[:num_samples]
    
    if len(images) == 0:
        return
    
    cols = 4
    rows = (len(images) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(images):
        row = idx // cols
        col = idx % cols
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[row, col].imshow(img_rgb)
        axes[row, col].set_title(img_path.name[:30], fontsize=8)
        axes[row, col].axis('off')
    
    for idx in range(len(images), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Sample grid saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize dataset statistics")
    parser.add_argument("--dataset_dir", type=str, default="processed_dataset")
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--num_samples", type=int, default=8)
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating dataset statistics...")
    plot_dataset_statistics(dataset_dir, output_dir / "dataset_statistics.png")
    
    print("Creating sample grid...")
    create_sample_grid(dataset_dir, args.num_samples, output_dir / "sample_grid.png")
    
    print("Visualization done!")


if __name__ == "__main__":
    main()

