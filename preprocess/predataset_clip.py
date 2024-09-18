from cProfile import label
import torch
import json
from tqdm import tqdm
from PIL import Image,ImageFile
import pickle
import numpy as np
from transformers import CLIPModel, CLIPProcessor
# from transformers import BlipModel, BlipProcessor
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

dataset_template = {
    'train': {
        "raw_text": [],
        "heu_prompt": [],
        "vision": [],
        "id": [],
        "text": [],
        "text_bert": [],
        "heu_prompt_lengths": [],
        "vision_lengths": [],
        "annotations": [],
        "classification_labels": [],
        "regression_labels": []  
    }
}



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model_name = 'openai/clip-vit-base-patch32'
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    
    caption_dir ='artphoto_captions.json'
    object_dir = 'artphoto_objects.json'
    place_dir = 'artphoto_place.json'
    label_dir = '~/SentiFormer/datasets/LDL/Twitter_label.json'
    removelist_dir = '~/SentiFormer/preprocess/remove.json'
    removelist = []
    with open(caption_dir, 'r') as f,open(object_dir, 'r') as objects, open(place_dir, 'r') as places,open(label_dir, 'r') as labels,open(removelist_dir,'r') as remove:
        captions = json.load(f)
        ob = json.load(objects)
        pl = json.load(places)
        la = json.load(labels)
        re = json.load(remove)
        for i,caption in tqdm(enumerate(captions)):
            sentence = caption['image_caption']
            image_id = caption['image_id']
            img_dir = "~/SentiFormer/datasets/testImages_artphoto/"+caption['image_id']
            # if img_dir in re:
            #     continue
            inputs = clip_processor(text=[sentence], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():  
                outputs = clip_model.get_text_features(**inputs)
            sentence_embedding = outputs.cpu()
            dataset_template['train']['text'].append(sentence_embedding)
            image = Image.open(img_dir)
            inputs = clip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = clip_model.get_image_features(**inputs)
            
            image_embedding = outputs.cpu()

            dataset_template['train']['vision'].append(image_embedding)
            place = eval(pl[i]["image_objects"])[0]
            object = [item.split(',')[0] for item in ob[i]['image_objects']]
            confidence =[item for item in ob[i]['image_objects_score']]
            prompt = 'the scene or background of image is '+place 
            prompt +=', and the image contains the following objects: '
            for o,c in zip(object,confidence):
                if c < 0.4:
                    continue
                prompt+=o+', '
            prompt = prompt.replace('_',' ')
            inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():  
                outputs = clip_model.get_text_features(**inputs)

            sentence_embedding = outputs.cpu()
            dataset_template['train']['heu_prompt'].append(sentence_embedding)
            # name = la[caption['image_id']]
            name = image_id.split('_')[0]
            dataset_template['train']['classification_labels'].append(name)

    with open("processed_data_artphoto.pkl", "wb") as file:
        pickle.dump(dataset_template, file)
