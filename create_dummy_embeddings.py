import numpy as np
import pickle
import os

# Paths
BASE_DIR = './data/coco/'
EMB_DIM = 1024  # Original StackGAN embedding dimension

def create_dummy_embeddings(split='train', num_images=1000):
    # Load filenames
    with open(os.path.join(BASE_DIR, split, 'filenames.pickle'), 'rb') as f:
        filenames = pickle.load(f)
    
    # Create random embeddings for each image
    # Each image gets 10 random embeddings as in original StackGAN
    embeddings = []
    for _ in range(len(filenames)):
        # Create 10 random embeddings for each image
        img_embeddings = np.random.normal(0, 0.02, size=(10, EMB_DIM))
        # Normalize embeddings
        img_embeddings = img_embeddings / np.linalg.norm(img_embeddings, axis=1, keepdims=True)
        embeddings.append(img_embeddings)
    
    # Save embeddings
    out_path = os.path.join(BASE_DIR, split, 'char-CNN-RNN-embeddings.pickle')
    with open(out_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"Created dummy embeddings for {len(filenames)} images in {split} set")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    # Create embeddings for both train and val sets
    create_dummy_embeddings('train', 1000)
    create_dummy_embeddings('val', 200) 