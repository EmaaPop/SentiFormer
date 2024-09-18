import pickle
from transformers import RobertaModel, RobertaTokenizer
import torch
import json
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rob_model_name = 'roberta-base'  
rob_tokenizer = RobertaTokenizer.from_pretrained(rob_model_name)
rob_model = RobertaModel.from_pretrained(rob_model_name).to(device)

with open('~/dataset/FI/process_code/output_hela.json', 'r') as objects, open('~/dataset/FI/process_code/places365_output.json', 'r') as places:
    ob = json.load(objects)
    pl = json.load(places)
    with open('processed_data.pkl', 'wb+') as file:
        data = pickle.load(file)
        for i in tqdm(range (len(ob))):
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
            data['train']['heu_prompt'].append(sentence_embedding)
            name = pl[i]['image_id'].split('_')[0]
            data['train']['classification_labels'].append(name)

        pickle.dump(data, file)
