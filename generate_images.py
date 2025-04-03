from PIL import Image
import os
import pandas as pd

df = pd.read_csv("chrispo_data.csv")
os.makedirs("images", exist_ok=True)

for filename in df["ImageFilename"]:
    img_path = os.path.join("images", filename)
    if not os.path.exists(img_path):
        img = Image.new("RGB", (200, 200), color="gray")
        img.save(img_path)

print("✅ Dummy images created in 'images/'")
