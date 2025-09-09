import os, random
from PIL import Image

src = 'data'      # Source directory containing raw images
dst = 'dataset'   # Destination directory for processed dataset
size = (224, 224) # Target image size for the model

def create_folders():
    """
    Create train, validation, and test folders with class subdirectories (Exp, Safe).
    """
    for split in ['train', 'validation', 'test']:
        for cls in ['Exp', 'Safe']:
            os.makedirs(f'{dst}/{split}/{cls}', exist_ok=True)

def get_images(folder):
    """
    Validate and filter images in a given folder:
      - Keep only images with minimum dimension >= 32
      - Remove corrupted or too-small images
    """
    valid = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            with Image.open(path) as img:
                img.verify()  # Verify image integrity
            with Image.open(path) as img:
                if min(img.size) >= 32:
                    valid.append(path)
                else:
                    os.remove(path)
        except:
            try: os.remove(path)
            except: pass
    return valid

def process_image(input_path, output_path):
    """
    Convert an image to RGB, resize to target size, 
    and save as JPEG with high quality.
    """
    try:
        with Image.open(input_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(output_path, 'JPEG', quality=95)
    except:
        pass

def split_and_save():
    """
    Split dataset into train (70%), validation (15%), test (15%)
    and save processed images into respective folders.
    """
    def split_list(data):
        random.shuffle(data)
        n = len(data)
        return data[:int(0.7*n)], data[int(0.7*n):int(0.85*n)], data[int(0.85*n):]

    for cls in ['Exp', 'Safe']:
        images = get_images(os.path.join(src, cls))
        train, val, test = split_list(images)
        for group, name in zip([train, val, test], ['train', 'validation', 'test']):
            out_dir = f'{dst}/{name}/{cls}'
            for img_path in group:
                new_name = os.path.splitext(os.path.basename(img_path))[0] + '_processed.jpg'
                process_image(img_path, os.path.join(out_dir, new_name))

def analyze():
    """
    Count number of images per split (train/val/test) and per class.
    Returns a dictionary with statistics.
    """
    stats = {}
    for split in ['train', 'validation', 'test']:
        stats[split] = {}
        total = 0
        for cls in ['Exp', 'Safe']:
            folder = f'{dst}/{split}/{cls}'
            count = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            stats[split][cls] = count
            total += count
        stats[split]['total'] = total
    return stats

# Run preprocessing
create_folders()
split_and_save()
