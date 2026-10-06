import zipfile
from pathlib import Path
import shutil

# Folder containing the three ZIP files
base = Path(__file__).parent

# ZIP files
zip_files = {
    "YOLOv8n": base / "YOLOv8n_finetuned_results.zip",
    "YOLOv10n": base / "YOLOv10n_finetuned_results.zip",
    "YOLO11n": base / "YOLO11n_finetuned_results.zip"
}

# Temporary extraction folder
extract_folder = base / "merged_results"

# Remove old folder if it exists
if extract_folder.exists():
    shutil.rmtree(extract_folder)

extract_folder.mkdir()

# Extract each ZIP into its own model folder
for model_name, zip_path in zip_files.items():

    if not zip_path.exists():
        print(f"ERROR: {zip_path.name} not found")
        continue

    model_folder = extract_folder / model_name
    model_folder.mkdir()

    print(f"Extracting {zip_path.name}...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(model_folder)

print("\nAll three ZIP files extracted.")

# Create final ZIP
final_zip = base / "Railway_Animal_Final_Results.zip"

if final_zip.exists():
    final_zip.unlink()

shutil.make_archive(
    str(final_zip.with_suffix("")),
    "zip",
    root_dir=extract_folder
)

print("\n======================================")
print("FINAL ZIP CREATED")
print("======================================")
print(final_zip)