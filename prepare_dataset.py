
from pathlib import Path
from collections import Counter
from huggingface_hub import hf_hub_download
import zipfile
import shutil

# Main project folder
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"

# Six selected tomato categories
CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot"
]

print("Preparing PlantVillage dataset...")

# Get the already downloaded original ZIP file
zip_path = hf_hub_download(
    repo_id="mohanty/PlantVillage",
    filename="data.zip",
    repo_type="dataset"
)

# Download or locate the official split lists
split_files = {}

for split in ["train", "test"]:
    split_files[split] = hf_hub_download(
        repo_id="mohanty/PlantVillage",
        filename=f"splits/color_{split}.txt",
        repo_type="dataset"
    )

# Count extracted images
image_counts = {
    "train": Counter(),
    "test": Counter()
}

total_copied = 0

# Read the ZIP without extracting the entire dataset
with zipfile.ZipFile(zip_path, "r") as archive:

    for split in ["train", "test"]:

        print(f"\nExtracting {split} images...")

        with open(split_files[split], "r", encoding="utf-8") as file:

            for line in file:
                archive_path = line.strip()

                if not archive_path:
                    continue

                parts = archive_path.split("/")

                if len(parts) < 4:
                    continue

                class_name = parts[2]

                if class_name not in CLASSES:
                    continue

                # Create the correct output folder
                output_folder = DATASET_DIR / split / class_name
                output_folder.mkdir(parents=True, exist_ok=True)

                output_path = output_folder / parts[-1]

                # Copy only the selected image
                with archive.open(archive_path) as source:
                    with open(output_path, "wb") as destination:
                        shutil.copyfileobj(source, destination)

                image_counts[split][class_name] += 1
                total_copied += 1

                if total_copied % 500 == 0:
                    print(f"Images extracted: {total_copied}")

# Display final results
print("\n========== EXTRACTION COMPLETED ==========")

for split in ["train", "test"]:
    print(f"\n{split.upper()} IMAGES")

    for class_name in CLASSES:
        print(
            f"{class_name}: "
            f"{image_counts[split][class_name]}"
        )

    print(
        f"Total {split} images: "
        f"{sum(image_counts[split].values())}"
    )

print(f"\nTotal images extracted: {total_copied}")
print("Dataset preparation finished!")
