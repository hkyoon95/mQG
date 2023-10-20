from pathlib import Path
import pandas as pd
# import neuralcoref
# import spacy
import os
import joblib

dir_path = os.path.dirname(os.path.realpath(__file__))

# nlp = spacy.load('en_core_web_lg')
# neuralcoref.add_to_pipe(nlp, greedyness=0.45)

counter = 0
counter_q = 0


data = {}
for split in ['train', 'val', 'test']:
    mapping_path = dir_path + '/data/ftqa_wh_2_{}.xlsx'.format(split)

    mapping = pd.read_excel(mapping_path, engine='openpyxl')
    split = split[:-1]
    # due to dataloader work
    # if len(mapping)%2 != 0:
    #     mapping.drop(len(mapping)-1, inplace=True)
    data[split] = {}
    unique_sentence = []

    for i in range(mapping.shape[0]):

        question_type = mapping.loc[i, 'wh']

        section = mapping.loc[i, 'cor_section']

        question = mapping.loc[i, 'question']
        # answer = mapping.loc[i, 'answer']

        

        if section not in unique_sentence:
            unique_sentence.append(section)
            data[split][section] = []
            
        data[split][section].append({'question': question, 'wh': question_type, 'section': section})
        counter_q +=1
    print(counter_q)        
print(counter_q)

joblib.dump(data, dir_path + '/data/ftqa_wh_2_data.pkl')