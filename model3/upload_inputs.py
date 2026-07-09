import os
import subprocess

INPUT_DIR = "input_data"

print("Checking input_data folder...")

files = [f for f in os.listdir(INPUT_DIR) if f != "dataset-metadata.json"]

if len(files) == 0:
    print(f"Error: The '{INPUT_DIR}' folder is empty!")
    print("Please place your video (.mp4) and query image (.jpeg/.png) inside the 'input_data' folder and run this script again.")
    exit(1)

print(f"Found files: {files}")
print("Uploading to Kaggle Dataset 'directorprince/forensic-input'...")

try:
    # Try to create it if it doesn't exist
    subprocess.run(["python", "-m", "kaggle", "datasets", "create", "-p", INPUT_DIR], check=True)
    print("Successfully created dataset on Kaggle!")
except subprocess.CalledProcessError:
    # If it already exists, create a new version
    print("Dataset likely already exists. Pushing new version...")
    subprocess.run(["python", "-m", "kaggle", "datasets", "version", "-p", INPUT_DIR, "-m", "Update inputs"], check=True)
    print("Successfully updated dataset on Kaggle!")

print("\nInputs are now on Kaggle!")
print("To use these inputs, make sure your Kaggle notebook reads from: /kaggle/input/forensic-input/")
