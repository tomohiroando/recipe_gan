import sys
from gensim import models
from gensim.models.doc2vec import LabeledSentence
import pickle


def corpus_to_sentences(corpus):
    sentences = []
    for idx, (name, doc) in enumerate(corpus.items()):
        sys.stdout.write('\r前処理中 {}/{}'.format(idx, len(corpus)))
        sentence = LabeledSentence(words=doc, tags=[name])
        sentences.append(sentence)

    return sentences


with open('corpus_text', 'rb') as f:
    corpus = pickle.load(f)

sentences = corpus_to_sentences(corpus)

model = models.Doc2Vec(vector_size=400, window=15, alpha=.025, min_alpha=.025, min_count=1, sample=1e-6)
model.build_vocab(sentences)
print(len(corpus))
model.train(sentences, total_examples=len(corpus), epochs=20)
model.save('doc2vec.model')