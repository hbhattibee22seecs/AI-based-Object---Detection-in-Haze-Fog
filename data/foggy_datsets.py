# datasets/foggy_dataset.py
import random, math, json, os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def apply_simple_fog(img, fog_strength=0.35):
    # img: HxWx3 uint8
    img = img.astype(np.float32)/255.0
    t = np.exp(-fog_strength)                # transmission
    A = np.random.uniform(0.7, 1.0)          # atmospheric light
    fogged = img * t + A * (1 - t)
    return (np.clip(fogged,0,1)*255).astype(np.uint8)

def mixup_examples(img1, target1, img2, target2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    img = (img1 * lam + img2 * (1-lam)).astype(np.uint8)
    # For detection: concatenate boxes/labels (paper uses Mixup and combined labels)
    boxes = np.concatenate([target1['boxes'], target2['boxes']], axis=0)
    labels = np.concatenate([target1['labels'], target2['labels']], axis=0)
    return img, {'boxes': boxes, 'labels': labels, 'mix_lambda': lam}

class FoggyCOCODataset(Dataset):
    def __init__(self, images_dir, coco_json_path, input_size=640, use_mixup=True, fog_prob=0.6):
        self.images_dir = images_dir
        with open(coco_json_path,'r') as f:
            data = json.load(f)
        # Build id->annotations map, and an index list mapping idx->(img_path, annotations)
        self.images = {img['id']: img for img in data['images']}
        anns = {}
        for a in data['annotations']:
            img_id = a['image_id']
            anns.setdefault(img_id, []).append(a)
        self.indexes = []
        for img_id, img_meta in self.images.items():
            self.indexes.append((img_meta['file_name'], anns.get(img_id, [])))
        self.input_size = input_size
        self.use_mixup = use_mixup
        self.fog_prob = fog_prob
        self.tfm = T.Compose([
            T.ToPILImage(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        file_name, ann_list = self.indexes[idx]
        img_path = os.path.join(self.images_dir, file_name)
        img = cv2.imread(img_path)[:,:,::-1]  # BGR->RGB
        if random.random() < self.fog_prob:
            fog_s = random.uniform(0.2, 0.6)   # vary fog density
            img = apply_simple_fog(img, fog_strength=fog_s)

        boxes = []
        labels = []
        for a in ann_list:
            x,y,w,h = a['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(a['category_id'])
        target = {'boxes': np.array(boxes, dtype=np.float32), 'labels': np.array(labels, dtype=np.int64)}

        # Mixup with probability 0.5 during training
        if self.use_mixup and random.random() < 0.5:
            idx2 = random.randint(0, len(self.indexes)-1)
            file_name2, ann_list2 = self.indexes[idx2]
            img2 = cv2.imread(os.path.join(self.images_dir, file_name2))[:,:,::-1]
            if random.random() < self.fog_prob:
                img2 = apply_simple_fog(img2, fog_strength=random.uniform(0.2,0.6))
            boxes2=[]; labels2=[]
            for a in ann_list2:
                x,y,w,h = a['bbox']
                boxes2.append([x, y, x+w, y+h])
                labels2.append(a['category_id'])
            img, target = mixup_examples(img, target, img2, {'boxes': np.array(boxes2), 'labels': np.array(labels2)})

        img_t = self.tfm(img)
        # YOLOX expects targets per its loader. We'll return raw boxes/labels; training script will convert
        return img_t, target

