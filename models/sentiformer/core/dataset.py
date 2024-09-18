import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import OneHotEncoder

categories = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]

categories_reshaped = [[cat] for cat in categories]

encoder = OneHotEncoder()


one_hot_encoded = encoder.fit_transform(categories_reshaped)
one_hot_encoded = one_hot_encoded.toarray()
name2id={
    'amusement': one_hot_encoded[0],
    'anger': one_hot_encoded[1],
    'awe': one_hot_encoded[2],
    'contentment': one_hot_encoded[3],
    'disgust': one_hot_encoded[4],
    'excitement': one_hot_encoded[5],
    'fear': one_hot_encoded[6],
    'sadness': one_hot_encoded[7]
}


class Meta(Dataset):
    def __init__(self, args):
        self.args = args
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)
        self.text = data['train']['text']
        self.heu_prompt = data['train']['heu_prompt']
        self.vision = data['train']['vision']
        self.labels = [name2id[n] for n in data['train']['classification_labels']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature1 = self.text[idx]
        feature2 = self.heu_prompt[idx]
        feature3 = self.vision[idx]
        label = self.labels[idx]
        
        features = torch.concat([feature1, feature2, feature3], dim=0)
        label = torch.tensor(label, dtype=torch.float32)
        
        return features, label


