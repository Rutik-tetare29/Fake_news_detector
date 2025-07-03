import gdown
import os

# Ensure 'data' folder exists
os.makedirs("data", exist_ok=True)

# Replace these with your actual file IDs
files = {
    "news.csv": "1S4c4L-Xd0zu0dd4yEgE3irxSaTYilNOH",
    "Fake.csv": "1vAFHRRSkyflUQnkecvcGaMLPCbwpp8qU",
    "True.csv": "1zj-CYUyfmScpE6Wvo20fTa8N4O6dOlOE"
}

# Download each file
for filename, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = os.path.join("data", filename)
    print(f"📥 Downloading {filename}...")
    gdown.download(url, output_path, quiet=False)

print("✅ All files downloaded to the 'data/' folder.")
