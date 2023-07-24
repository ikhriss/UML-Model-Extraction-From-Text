import random
import numpy as np
import os
import sys
from transformers import BertTokenizer
import torch
from torch.autograd import Variable
import json
import utils 


def prepare_index():
    mentions_index = {"O": 0, "H": 1, "R": 2}
    embeddings_index = {}

    fichier = open("data/glove.6B/glove.6B.50d.txt", 'r', encoding="UTF-8")
    for ligne in fichier.readlines():
        rangee = ligne.strip().split(' ')
        embeddings_index[rangee[0]] = [float(i) for i in rangee[1:]]
    fichier.close()

    return embeddings_index, mentions_index


def prepare_document(entry, embeddings_index, mentions_index):
    document = Document()
    sentence = Sentence()
    iterator = 0

    file = open(entry, 'r', encoding="UTF-8")
    for line in file.readlines():
        row = line.strip().split(' ')
        if len(row) < 2:
            document.sentences.append(sentence)
            sentence = Sentence()
        else:
            wording = row[0].lower()
            position = iterator
            embedding = torch.cuda.FloatTensor([embeddings_index.get(row[0].lower(), embeddings_index.get("unk"))])            
            sentence.words.append(Word(wording, embedding, position))
            sentence.mention_goldens = torch.cat((sentence.mention_goldens, torch.cuda.LongTensor([mentions_index.get(row[1])])), dim=0)
            if len(row) > 2:
                sentence.coreference_goldens = torch.cat((sentence.coreference_goldens, torch.cuda.LongTensor([[int(row[2])]])), dim=0)
            if len(row) > 3:
                row_relations = row[3].split('-')
                for row_relation in row_relations:
                    sentence.relation_goldens = torch.cat((sentence.relation_goldens, torch.cuda.LongTensor([int(row_relation)])), dim=0)
            iterator += 1
    
    document.sentences.append(sentence)

    return document


def prepare_corpus(prefixe, entries, embeddings_index, mentions_index):
    corpus = Corpus()

    for entry in entries:
        document = prepare_document(prefixe + entry, embeddings_index, mentions_index)
        corpus.documents.append(document)

    return corpus


def prepare_corpora(embeddings_index, mentions_index):
    train_entries = os.listdir('data/train')
    val_entries = os.listdir('data/val')
    test_entries = os.listdir('data/test')

    train_corpus = prepare_corpus('data/train/', train_entries, embeddings_index, mentions_index)
    val_corpus = prepare_corpus('data/val/', val_entries, embeddings_index, mentions_index)
    test_corpus = prepare_corpus('data/test/', test_entries, embeddings_index, mentions_index)

    return train_corpus, val_corpus, test_corpus


class Corpus:
    def __init__(self):
        self.documents = []
        

class Document:
    def __init__(self):
        self.sentences = []
        self.clusters = []
    
    def show_document(self):
        print("Document:")
        for sentence in self.sentences:
            print("Sentence:", " ".join(word.wording for word in sentence.words))
            for mention in sentence.mentions:
                print("Mention:", " ".join(word.wording for word in mention.words))
        print("Clusters:")
        for cluster in self.clusters:
            print("Cluster:")
            for mention in cluster.mentions:
                print("Mention:", " ".join(word.wording for word in mention.words))
                print("Cluster ID:", mention.cluster)
                print("Head:", mention.head)

    def save(self, file_name):
      with open(file_name, "w", encoding="utf-8") as file:
            file.write("Document:\n")
            for sentence in self.sentences:
                file.write("Sentence: " + " ".join(word.wording for word in sentence.words) + "\n")
                for mention in sentence.mentions:
                    file.write("Mention: " + " ".join(word.wording for word in mention.words) + "\n")
            file.write("Clusters:\n")
            for cluster in self.clusters:
                file.write("Cluster:\n")
                for mention in cluster.mentions:
                    file.write("Mention: " + " ".join(word.wording for word in mention.words) + "\n")
                    file.write("Cluster ID: " + str(mention.cluster) + "\n")
                    file.write("Head: " + str(mention.head) + "\n")
class Sentence:
    def __init__(self):
        self.words = []
        self.mentions = []
        self.mention_goldens = torch.cuda.LongTensor()
        self.coreference_goldens = torch.cuda.LongTensor()
        self.relation_goldens = torch.cuda.LongTensor()


class Word:
    def __init__(self, wording, embedding, position):
        self.wording = wording
        self.embedding = embedding
        self.position = position
        self.encoding = torch.cuda.FloatTensor()


class Mention:
    def __init__(self, word, head=-1):
        self.words = [word]
        self.cluster = 0
        self.head = head
        self.attentionLayer = torch.cuda.FloatTensor()


class Cluster:
    def __init__(self, mention):
        self.mentions = [mention]
        self.relations = {2: [], 3: [], 4: []} # relations par type avec les autres clusters {2: [], 3: [], 4: [], ...}
