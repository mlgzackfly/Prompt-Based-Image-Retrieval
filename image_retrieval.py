import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from scipy.spatial.distance import cdist

class CFG:
    model_path = 'model/vit_base_patch16_224.pth'
    model_name = 'vit_base_patch16_224'
    input_size = 224
    batch_size = 64


class DiffusionTestDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])

        # 裁剪圖像以使所有圖像具有相同的形狀
        w, h = image.size
        if w > h:
            left = (w - h) // 2
            right = left + h
            top, bottom = 0, h
        else:
            top = (h - w) // 2
            bottom = top + w
            left, right = 0, w
        image = image.crop((left, top, right, bottom))

        image = self.transform(image)
        return image


def predict(images, model_path, model_name, input_size, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = DiffusionTestDataset(images, transform)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0,
        drop_last=False
    )

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=384
    )
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    preds = []
    for X in tqdm(dataloader, leave=False):
        X = X.to(device)
        with torch.no_grad():
            X_out = model(X)
            preds.append(X_out.cpu().numpy())

    return np.vstack(preds)

# 提取所有圖片的 prompt_embeddings
image_dir = Path('kaggle')
image_files = list(image_dir.glob('*.[pP][nN][gG]'))
image_embeddings = predict(image_files, CFG.model_path, CFG.model_name, CFG.input_size, CFG.batch_size)
image_ids = [f.stem for f in image_files]
image_embeddings_df = pd.DataFrame(
    index=image_ids,
    data=image_embeddings
)

# 正規化
image_embeddings_norm = np.linalg.norm(image_embeddings_df, axis=1, keepdims=True)
image_embeddings_normed = image_embeddings_df / image_embeddings_norm

# 計算圖像之間的歐式距離
dist_mat = cdist(image_embeddings_normed, image_embeddings_normed, metric='euclidean')

print(image_ids)

query_image_id = 'example'
query_image_index = image_ids.index(query_image_id)

# 取得與查詢圖像最相似的前 5 張圖像及其相似度
n_results = 5
distances = dist_mat[query_image_index]
top_indices = np.argsort(distances)[:n_results]
top_distances = distances[top_indices]

# 顯示結果
print(f'Top {n_results} similar images to {query_image_id}:')
for i, (idx, distance) in enumerate(zip(top_indices, top_distances)):
    print(f'Top {i + 1}: {image_ids[idx]} (similarity: {1 / (distance + 1e-6):.2f})')
