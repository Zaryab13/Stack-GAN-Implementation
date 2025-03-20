import os
import requests
import json
import shutil
from tqdm import tqdm
import pickle
import zipfile

# Create data directories
BASE_DIR = './data/coco/'
DATA_DIR = './data/coco2014/'
TRAIN_DATA = 'train2014'
VAL_DATA = 'val2014'

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, TRAIN_DATA), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, VAL_DATA), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'val'), exist_ok=True)

# Number of images to download
NUM_TRAIN_IMAGES = 1000  # Instead of ~80K
NUM_VAL_IMAGES = 200    # Instead of ~40K

def download_image(img_url, save_path):
    try:
        response = requests.get(img_url, stream=True)
        if response.status_code == 200:
            # Get the total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Create a progress bar for this download
            with open(save_path, 'wb') as f, tqdm(
                desc=os.path.basename(save_path),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            return True
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")
        return False
    return False

# Download COCO annotations
print('Downloading COCO annotations...')
ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
ann_file = os.path.join(BASE_DIR, 'annotations.zip')

# Show progress for annotations download
response = requests.get(ann_url, stream=True)
total_size = int(response.headers.get('content-length', 0))
with open(ann_file, 'wb') as f, tqdm(
    desc="annotations.zip",
    total=total_size,
    unit='B',
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
            bar.update(len(chunk))

# Show progress for extraction
print('Extracting annotations...')
with zipfile.ZipFile(ann_file, 'r') as zip_ref:
    # Get list of files to extract for progress tracking
    file_list = zip_ref.namelist()
    for file in tqdm(file_list, desc="Extracting"):
        zip_ref.extract(file, BASE_DIR)
        
os.remove(ann_file)

# Load annotations
print('Loading annotations...')
with open(os.path.join(BASE_DIR, 'annotations/instances_train2014.json')) as f:
    train_anns = json.load(f)
with open(os.path.join(BASE_DIR, 'annotations/instances_val2014.json')) as f:
    val_anns = json.load(f)

# Download subset of training images
print(f'Downloading {NUM_TRAIN_IMAGES} training images...')
train_images = train_anns['images'][:NUM_TRAIN_IMAGES]
for img in tqdm(train_images, desc="Training images"):
    file_name = img['file_name']
    img_url = f'http://images.cocodataset.org/train2014/{file_name}'
    save_path = os.path.join(DATA_DIR, TRAIN_DATA, file_name)
    if not os.path.exists(save_path):
        download_image(img_url, save_path)

# Download subset of validation images
print(f'Downloading {NUM_VAL_IMAGES} validation images...')
val_images = val_anns['images'][:NUM_VAL_IMAGES]
for img in tqdm(val_images, desc="Validation images"):
    file_name = img['file_name']
    img_url = f'http://images.cocodataset.org/val2014/{file_name}'
    save_path = os.path.join(DATA_DIR, VAL_DATA, file_name)
    if not os.path.exists(save_path):
        download_image(img_url, save_path)

print('Dataset download complete!')

# Create pickle files with filenames
print('Creating pickle files...')
train_filenames = [img['file_name'] for img in train_images]
val_filenames = [img['file_name'] for img in val_images]

with open(os.path.join(BASE_DIR, 'train/filenames.pickle'), 'wb') as f:
    pickle.dump(train_filenames, f)
with open(os.path.join(BASE_DIR, 'val/filenames.pickle'), 'wb') as f:
    pickle.dump(val_filenames, f)

print('Setup complete!')