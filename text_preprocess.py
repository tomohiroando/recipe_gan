from janome.tokenizer import Tokenizer
import pandas as pd
import pickle

recipe_dict = {}
stopwords = ['する', 'なる']
t = Tokenizer()

f = open('./recipe_20170118/recipe01_all_20170118.txt')

df_txtdata = pd.read_table('./recipe_20170118/recipe03_process_20160112.txt', names=('id', 'num', 'sentence'))
print(df_txtdata['id'].head(3))


line = f.readline()
while line:
    list = []
    recipe = line.split("\t")
    if(recipe[2] != '卵料理'):
        line = f.readline()
        continue
    recipe_id = recipe[0]
    df_extract = df_txtdata[df_txtdata['id'] == int(recipe_id)]

    for sentence in df_extract['sentence'].dropna():
        for token in t.tokenize(sentence):
            if(token.part_of_speech.split(',')[0] == '名詞') and (token.part_of_speech.split(',')[1] == '一般') and(token.surface.isalnum() != True):
                list.append(token.surface)
            if(token.part_of_speech.split(',')[0] == '動詞') and (token.base_form not in stopwords):
                list.append(token.base_form)
    recipe_dict[recipe_id] = list

    line = f.readline()



f.close()
print(recipe_dict)
f = open('corpus_text', 'wb')
pickle.dump(recipe_dict, f)
f.close()