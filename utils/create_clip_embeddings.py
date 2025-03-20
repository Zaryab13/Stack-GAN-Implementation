import torch
import clip
from PIL import Image
import os
import pickle
import numpy as np
from tqdm import tqdm

# Paths
BASE_DIR = './data/coco/'
DATA_DIR = './data/coco2014/'
TRAIN_DATA = 'train2014'
VAL_DATA = 'val2014'

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)
    return model, preprocess, device

def create_clip_embeddings(split='train'):
    print(f"Creating CLIP embeddings for {split} set...")
    
    # Load CLIP model
    model, preprocess, device = load_clip_model()
    model.eval()
    
    # Load filenames
    with open(os.path.join(BASE_DIR, split, 'filenames.pickle'), 'rb') as f:
        filenames = pickle.load(f)
    
    # Get image directory
    img_dir = os.path.join(DATA_DIR, f'{split}2014')
    
    # Generate embeddings
    embeddings = []
    successful_files = 0
    
    for filename in tqdm(filenames):
        try:
            # Construct image path for COCO - don't add .jpg as it's already in the filename
            image_path = os.path.join(img_dir, filename)
            
            # If the file doesn't exist at the exact path, try some variations
            if not os.path.exists(image_path):
                # Try removing .jpg if it exists and add it again (handles potential .jpg.jpg issues)
                if filename.lower().endswith('.jpg'):
                    base_name = filename[:-4]  # Remove .jpg
                    image_path = os.path.join(img_dir, base_name + '.jpg')
                # If still doesn't exist, try without extension
                if not os.path.exists(image_path) and not filename.lower().endswith('.jpg'):
                    image_path = os.path.join(img_dir, filename + '.jpg')
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found after trying variations - {image_path}")
                continue
                
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            
            # Generate multiple embeddings with different augmentations
            img_embeddings = []
            with torch.no_grad():
                # Get base embedding
                base_embedding = model.encode_image(image).cpu().numpy()
                
                # Create 10 variations by adding small random noise
                for _ in range(10):
                    noise = np.random.normal(0, 0.02, base_embedding.shape)
                    variant = base_embedding + noise
                    # Normalize
                    variant = variant / np.linalg.norm(variant)
                    img_embeddings.append(variant)
            
            embeddings.append(np.stack(img_embeddings))
            successful_files += 1
            
        except Exception as e:
            print(f"Error processing image {filename}: {e}")
    
    # Save embeddings
    out_path = os.path.join(BASE_DIR, split, 'char-CNN-RNN-embeddings.pickle')
    with open(out_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"Created CLIP embeddings for {successful_files}/{len(filenames)} images")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    # First install CLIP if not already installed
    # !pip install git+https://github.com/openai/CLIP.git
    
    # Create embeddings for both train and val sets
    create_clip_embeddings('train')
    create_clip_embeddings('val') 