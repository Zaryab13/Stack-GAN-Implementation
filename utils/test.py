import os
import pickle

img_dir = './data/coco2014/train2014/'
filenames_file = './data/coco/train/filenames.pickle'

with open(filenames_file, 'rb') as f:
    filenames = pickle.load(f)

print(f"Total filenames: {len(filenames)}")
for filename in filenames[:5]:  # Check the first 5
    image_path = os.path.join(img_dir, f'{filename}.jpg')
    if os.path.exists(image_path):
        print(f"Found: {image_path}")
    else:
        print(f"Not found: {image_path}")