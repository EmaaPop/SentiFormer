from transformers import RobertaModel, RobertaTokenizer
import torch
import json
from tqdm import tqdm
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import pickle
import numpy as np

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



from skimage.metrics import structural_similarity as ssim
import cv2


def resize_to_same_size(img1, img2):
    min_width = min(img1.width, img2.width)
    min_height = min(img1.height, img2.height)

    img1_resized = img1.resize((min_width, min_height))
    img2_resized = img2.resize((min_width, min_height))

    return img1_resized, img2_resized

def images_are_similar_ssim(image1_path, image2_path, threshold=0.9):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    img1, img2 = resize_to_same_size(img1, img2)

    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

    score, _ = ssim(img1, img2, full=True)
    return score >= threshold


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rob_model_name = 'roberta-base'  
    rob_tokenizer = RobertaTokenizer.from_pretrained(rob_model_name)
    rob_model = RobertaModel.from_pretrained(rob_model_name).to(device)
    
    vit_model_name = "google/vit-base-patch16-224"
    vit_model = ViTModel.from_pretrained(vit_model_name).to(device)
    vit_processor = ViTImageProcessor.from_pretrained(vit_model_name)
    
    caption_dir ='~/dataset/FI/FI_captions.json'
    removelist = []
    with open(caption_dir, 'r') as f,open('remove.json','r') as remove:
        captions = json.load(f)
        re = json.load(remove)
        for i,caption in tqdm(enumerate(captions)):
            sentence = caption['image_caption']
            img_dir = "~/dataset/FI/images/"+caption['image_id']
            if img_dir in re:
                continue
            inputs = rob_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():  
                outputs = rob_model(**inputs)

            last_hidden_state = outputs.last_hidden_state
            sentence_embedding = last_hidden_state[:, 0, :]
            dataset_template['train']['text'].append(sentence_embedding)
            image = Image.open(img_dir)

            inputs = vit_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = vit_model(**inputs)

            last_hidden_state = outputs.last_hidden_state
            
            image_embedding = last_hidden_state[:, 0, :]

            dataset_template['train']['vision'].append(image_embedding)
            with open('~/dataset/FI/process_code/output_hela.json', 'r') as objects, open('~/dataset/FI/process_code/places365_output.json', 'r') as places:
                ob = json.load(objects)
                pl = json.load(places)
                place = eval(pl[i]["image_places"])[0]
                object = [item.split(',')[0] for item in eval(ob[i]['image_objects'])]
                prompt = 'the place of image is '+place +'it contains following objects:'
                for o in object:
                    prompt+=o+','
                prompt = prompt.replace('_',' ')
                inputs = rob_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():  
                    outputs = rob_model(**inputs)

                last_hidden_state = outputs.last_hidden_state
                sentence_embedding = last_hidden_state[:, 0, :]
                dataset_template['train']['heu_prompt'].append(sentence_embedding)
                name = pl[i]['image_id'].split('_')[0]
                dataset_template['train']['classification_labels'].append(name)

    with open("processed_data.pkl", "wb") as file:
        pickle.dump(dataset_template, file)


            



    