import pickle

with open('./corpus_text', 'rb') as f:
    dictfile = pickle.load(f)

with open('./noexist_list', 'rb') as f:
    noexist_list = pickle.load(f)

listfile = []

for key in dictfile:
    if key not in noexist_list:
        listfile.append(key)

print(len(listfile))
f = open('corpus_index', 'wb')
pickle.dump(listfile, f)