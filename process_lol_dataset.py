from pathlib import Path
from dataset_preprocessing import LowLightDatasetPreprocessor
import argparse
import cv2


def process_lol_dataset(lol_dir: str = "lol_dataset", output_dir: str = "processed_dataset", 
                        use_high_as_ground_truth: bool = False):
    lol_path = Path(lol_dir)
    
    if not lol_path.exists():
        print(f"Error: {lol_dir} not found!")
        return
    
    print("="*70)
    print("Processing LOL Dataset")
    print("="*70)
    
    our485_low = lol_path / "our485" / "low"
    eval15_low = lol_path / "eval15" / "low"
    
    if not our485_low.exists():
        print(f"Error: {our485_low} not found!")
        return
    
    print(f"\nDataset structure detected:")
    print(f"  Training set: {our485_low} ({len(list(our485_low.glob('*.png')))} images)")
    print(f"  Evaluation set: {eval15_low} ({len(list(eval15_low.glob('*.png')))} images)")
    
    print("\n" + "="*70)
    print("Processing Training Set (our485)")
    print("="*70)
    
    preprocessor_train = LowLightDatasetPreprocessor(
        output_dir=output_dir,
        target_size=(512, 512)
    )
    
    stats_train = preprocessor_train.process_dataset(
        input_dir=str(our485_low),
        split_ratio=0.8,
        normalize=True,
        augment=True
    )
    
    print("\n" + "="*70)
    print("Processing Evaluation Set (eval15) as Test Set")
    print("="*70)
    
    test_dir = Path(output_dir) / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    eval_images = list(eval15_low.glob("*.png"))
    print(f"Processing {len(eval_images)} evaluation images...")
    
    for img_path in eval_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        resized = preprocessor_train.resize_image(img)
        normalized = preprocessor_train.normalize_image(resized, method="standard")
        cv2.imwrite(str(test_dir / f"eval_{img_path.name}"), normalized)
    
    print(f"Added {len(eval_images)} evaluation images to test set")
    
    print("\n" + "="*70)
    print("Creating Visual Examples")
    print("="*70)
    
    preprocessor_train.create_visual_examples(num_examples=5, source="train")
    preprocessor_train.create_visual_examples(num_examples=5, source="test")
    
    print("\n" + "="*70)
    print("Processing Done!")
    print("="*70)
    print(f"\nDataset Statistics:")
    print(f"  Training images (processed): {stats_train['processed_train']}")
    print(f"  Validation images (processed): {stats_train['processed_test']}")
    print(f"  Test images (eval15): {len(eval_images)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - train/ : Training images with augmentation")
    print(f"  - test/  : Test images (eval15) + validation split")
    print(f"  - visualizations/ : Example visualizations")
    
    if use_high_as_ground_truth:
        print("\nNote: Ground truth (high) images are available in:")
        print(f"  - {lol_path / 'our485' / 'high'}")
        print(f"  - {lol_path / 'eval15' / 'high'}")


def main():
    parser = argparse.ArgumentParser(description="Process LOL Dataset")
    parser.add_argument("--lol_dir", type=str, default="lol_dataset")
    parser.add_argument("--output_dir", type=str, default="processed_dataset")
    parser.add_argument("--target_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--use_high", action="store_true")
    
    args = parser.parse_args()
    
    if args.target_size != [512, 512]:
        print(f"Note: Using default size 512x512. Custom size requires code modification.")
    
    process_lol_dataset(
        lol_dir=args.lol_dir,
        output_dir=args.output_dir,
        use_high_as_ground_truth=args.use_high
    )


if __name__ == "__main__":
    main()

