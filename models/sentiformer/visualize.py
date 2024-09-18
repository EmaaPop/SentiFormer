import os
import torch
from tqdm import tqdm
from opts import *
from core.dataset import MMDataLoader
from core.utils import AverageMeter
from model.met.met import build_model
from core.metric import MetricsTop
from core.new_dataset import Meta
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.font_manager as fm
import pickle

opt = parse_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_VISIBLE_DEVICES
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("device: {}:{}".format(device, opt.CUDA_VISIBLE_DEVICES))


train_mae, val_mae = [], []



def main():
    opt = parse_opts()

    model = build_model(opt).to(device)
    model.load_state_dict(torch.load('SentiFormer/checkpoint/best/797.pth')['state_dict'])

    dataset = Meta(opt)
    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)
    val_dataset = Subset(dataset, val_indices)

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    data_list = []
    label_list = []

    for data, labels in val_loader:
        data_list.append(data)
        labels = torch.argmax(labels,dim=-1)
        label_list.append(labels)
        # print(labels.shape)
        # exit(0)


    data_tensor = torch.cat(data_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    feature = []
    with torch.no_grad():
        for data,labels in tqdm(val_loader):
            img, heu_prompt, text = data[:,2].to(device), data[:,1].to(device), data[:,0].to(device)
            _,r = model(img, heu_prompt, text)
            feature.append(r)
    features = torch.cat(feature,dim=0)
    features_np = features.view(features.size(0), -1).cpu().numpy()
    data_np = data_tensor.view(features.size(0), -1).cpu().numpy()
    with open('data_tensor.pkl', 'wb') as f:
        pickle.dump(data_np, f)
    with open('label_list.pkl', 'wb') as f:
        pickle.dump(label_list.cpu().numpy(), f)
    with open('features_np.pkl', 'wb') as f:
        pickle.dump(features_np, f)



    # 3D t-SNE
    
    tsne_3d = TSNE(n_components=3, random_state=0)
    features_3d = tsne_3d.fit_transform(features_np)
    categories = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
    plt.figure(figsize=(10, 7))
    tsne_2d = TSNE(n_components=2, random_state=0)
    features_2d = tsne_2d.fit_transform(data_np)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=label_list, cmap='viridis')
    # cbar = plt.colorbar(ticks=range(len(categories)))
    # cbar.set_ticklabels(categories)


    # 3D t-SNE
    # tsne_3d = TSNE(n_components=3, random_state=0)
    # features_3d = tsne_3d.fit_transform(data_np)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))


    axs[0].scatter(features_2d[:, 0], features_2d[:, 1], c=label_list, cmap='viridis')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    features_2d = tsne_2d.fit_transform(features_np)

    axs[1].scatter(features_2d[:, 0], features_2d[:, 1], c=label_list.cpu().numpy(), cmap='viridis')
    axs[1].set_xticks([])
    axs[1].set_yticks([])


    cbar = fig.colorbar(scatter, ax=axs, ticks=range(len(categories)), orientation='horizontal',pad=0.05)
    cbar.ax.set_aspect(0.02)

    cbar.set_ticklabels(categories)
    cbar.ax.tick_params(labelsize=26, labelrotation=45)



    plt.savefig('2dvisual.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':
    main()