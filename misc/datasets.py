import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import pickle
import numpy as np
from gensim import models


class RecipeDataset(data.Dataset):
    def __init__(self, data_dir, split='', imsize=64, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.imsize = imsize
        self.data = []
        self.data_dir = data_dir

        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, self.filenames, 'doc2vec.model')
        # corpus_text--> ['ファイル名', '単語リスト']なので，keyに対応する分散表現を取得

    def get_img(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
        except OSError:
            # 画像が破損しているときは代替画像を挟む
            img = Image.open('./img_data/1150006760.jpg').convert('RGB')
        width, height = img.size
        load_size = int(self.imsize * 76 / 64)
        img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img

    # 分散表現読み込み
    def load_embedding(self, data_dir, filenames, embedding_filename):
        doc_model = models.Doc2Vec.load(data_dir + embedding_filename)
        embeddings = np.array([doc_model.docvecs[key] for key in filenames])
        print('embeddings: ', embeddings.shape)
        return embeddings


    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'corpus_index')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def __getitem__(self, index):
        key = self.filenames[index]
        data_dir = self.data_dir

        embeddings = self.embeddings

        # 画像セット
        img_name = '%s/%s.jpg' % (data_dir, key)
        img = self.get_img(img_name)

        # 分散表現セット
        embedding = embeddings[index, :]
        return img, embedding

    def __len__(self):
        return len(self.filenames)