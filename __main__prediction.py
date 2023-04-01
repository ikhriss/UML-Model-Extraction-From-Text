import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import bcubed
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
from math import sqrt
# from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import classification_report


def cudaize(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


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


class DistanceLayer(nn.Module):
    # Encoding de distance
    def __init__(self, distance_dim, dropout):
        super().__init__()
        self.refs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        self.distance_dim = distance_dim
        self.layer = nn.Sequential(
            nn.Embedding(len(self.refs)+1, self.distance_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, distance):
        return self.layer(self._lookup(distance))

    def _lookup(self, distance):
        for i in range(len(self.refs)-1, -1, -1):
            if distance >= self.refs[i]:
                return torch.cuda.LongTensor([i])


class EncodingRNNLayer(nn.Module):
    # BiLSTM
    def __init__(self, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensor):
        output, _ = self.lstm(tensor.view(len(tensor), 1, -1))
        return self.dropout(output).view(output.size(0), output.size(2))


class EncodingBERTLayer(nn.Module):
    # BERT
    def __init__(self, dropout):
        super().__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = cudaize(BertModel.from_pretrained('bert-base-uncased'))
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence):
        word_sentence = [word.wording for word in sentence]
        inputs = self.bert_tokenizer.encode(word_sentence)
        outputs = self.bert(input_ids=cudaize(torch.tensor(inputs).view(-1, len(inputs))), attention_mask=cudaize(torch.ones(1, len(inputs), dtype=torch.long)), token_type_ids=cudaize(torch.zeros(1, len(inputs), dtype=torch.long)))
        return self.dropout(outputs.last_hidden_state[0][1:-1])


class EncodingGPTLayer(nn.Module):
    # GPT
    def __init__(self, dropout):
        super().__init__()
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt = cudaize(GPT2Model.from_pretrained('gpt2'))
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence):
        word_sentence = [word.wording for word in sentence]
        inputs = self.gpt_tokenizer.encode(word_sentence)
        outputs = self.gpt(input_ids=cudaize(torch.tensor(inputs).view(-1, len(inputs))), attention_mask=cudaize(torch.ones(1, len(inputs), dtype=torch.long)), token_type_ids=cudaize(torch.zeros(1, len(inputs), dtype=torch.long)))
        return self.dropout(outputs.last_hidden_state[0])


class ScoreLayer(nn.Module):
    # Classification
    def __init__(self, input_dim, hidden_dim, target_dim, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim)
        )

    def forward(self, tensor):
        return self.layer(tensor)


class ScoreTLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_dim, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim)
        )

    def forward(self, tensor):
        output = self.layer(tensor)
        return torch.transpose(output, 0, 1)


class Model(nn.Module):
    def __init__(self, method, technique, mention_loss_function, coreference_loss_function, relation_loss_function, embedding_dim, encoding_hidden_dim, decoding_hidden_dim, mention_dim, relation_dim, distance_dim, dropout):
        super().__init__()
        self.mode = 'training'
        self.method = method
        self.technique = technique
        self.mention_loss_function = mention_loss_function
        self.coreference_loss_function = coreference_loss_function
        self.relation_loss_function = relation_loss_function
        self.embedding_dim = embedding_dim
        self.encoding_hidden_dim = encoding_hidden_dim
        self.decoding_hidden_dim = decoding_hidden_dim
        self.mention_dim = mention_dim
        self.relation_dim = relation_dim
        self.distance_dim = distance_dim
        self.padding_dim_coef = 1
        if self.technique == 'padding':
            self.padding_dim_coef = 7   # Max size padding  
                      
        if self.method == 'RNN':
            self.encodingLayer = EncodingRNNLayer(self.embedding_dim, self.encoding_hidden_dim, dropout)
            self.encoding_hidden_dim = 2*encoding_hidden_dim    # Bilateral
        elif self.method == 'BERT':             
            self.encodingLayer = EncodingBERTLayer(dropout)
        else:
            self.encodingLayer = EncodingGPTLayer(dropout)
        self.mentionScoreLayer = ScoreLayer(self.encoding_hidden_dim, self.decoding_hidden_dim, self.mention_dim, dropout)
        self.attentionLayer = ScoreLayer(self.encoding_hidden_dim, self.decoding_hidden_dim, 1, dropout)
        self.coreferenceDistanceLayer = DistanceLayer(self.distance_dim, dropout)
        self.coreferenceScoreTLayer = ScoreTLayer(self.encoding_hidden_dim*3*self.padding_dim_coef + self.distance_dim, self.decoding_hidden_dim, 1, dropout)
        self.relationDistanceLayer = DistanceLayer(self.distance_dim, dropout)
        self.relationScoreLayer = ScoreLayer(self.encoding_hidden_dim*3*self.padding_dim_coef + self.distance_dim, self.decoding_hidden_dim, self.relation_dim, dropout)

    def forward(self, sentence, clusters, mode='training'):
        self.mode = mode  
        inputs = sentence.words
        if self.method == 'RNN':
            sentence_embeddings = torch.cuda.FloatTensor()
            for word in sentence.words:
                sentence_embeddings = torch.cat((sentence_embeddings, word.embedding), dim=0)            
            inputs = sentence_embeddings
                
        encodings = self.encodingLayer(inputs)
        for i in range(encodings.size(0)):
            sentence.words[i].encoding = encodings[i].view(1, encodings.size(1))

        # Mentions - Minibatch = sentence        
        mention_scores = self.mentionScoreLayer(encodings.view(len(sentence.words), -1))
        mention_scores_max_index = torch.max(mention_scores, dim=1).indices
        mention_loss = 0
        if (self.mode == 'training'):
            mention_loss = self.mention_loss_function(mention_scores, sentence.mention_goldens)
            if self.training:
                mention_loss.backward(retain_graph=True)   
        self._extract_mentions(sentence, mention_scores_max_index)

        # Coreferences - Minibatch = 1    
        coreference_total_scores_max_index = torch.cuda.LongTensor()
        coreference_total_loss = 0
        for i in range(len(sentence.mentions)):
            coreference_input = torch.zeros((1, 3*self.encoding_hidden_dim*self.padding_dim_coef + self.distance_dim), device=torch.device('cuda'))  # epsilon
            mention_attention = sentence.mentions[i].attention
            mention_position = sentence.mentions[i].words[0].position
            for cluster in clusters:
                cluster_attention = cluster.mentions[0].attention
                cluster_last_position = cluster.mentions[-1].words[0].position
                distance = self.coreferenceDistanceLayer(mention_position - cluster_last_position)
                new_input = torch.cat((mention_attention, cluster_attention.detach(), torch.mul(mention_attention, cluster_attention.detach()), distance), dim=1)
                coreference_input = torch.cat((coreference_input, new_input), dim=0)

            coreference_scores = self.coreferenceScoreTLayer(coreference_input)
            coreference_scores_max_index = torch.max(coreference_scores, dim=1).indices
            coreference_total_scores_max_index = torch.cat((coreference_total_scores_max_index, coreference_scores_max_index.view(1, 1)), dim=1)
            if (self.mode == 'training'):
                coreference_loss = self.coreference_loss_function(coreference_scores, sentence.coreference_goldens[i])
                coreference_total_loss += coreference_loss
                if self.training:
                    coreference_loss.backward(retain_graph=True)
            self._extract_clusters(sentence, clusters, i, coreference_scores_max_index)

        # Relations - Minibatch = sentence
        relation_input = torch.cuda.FloatTensor()
        relation_scores_max_index = torch.cuda.LongTensor()
        relation_loss = 0
        for i in range(len(sentence.mentions)):
            mention_attention = sentence.mentions[i].attention
            mention_position = sentence.mentions[i].words[0].position
            for j in range(i):
                previous_mention_attention = sentence.mentions[j].attention
                previous_mention_position = sentence.mentions[j].words[0].position
                distance = self.relationDistanceLayer(mention_position - previous_mention_position)
                new_input = torch.cat((mention_attention, previous_mention_attention, torch.mul(mention_attention, previous_mention_attention), distance), dim=1)
                relation_input = torch.cat((relation_input, new_input), dim=0)

        if relation_input.size()[0] > 0:
            relation_scores = self.relationScoreLayer(relation_input)
            relation_scores_max_index = torch.max(relation_scores, dim=1).indices
            if (self.mode == 'training'):
                relation_loss = self.relation_loss_function(relation_scores, sentence.relation_goldens)
                if self.training:
                    relation_loss.backward(retain_graph=True)   
            self._extract_relations(sentence, clusters, relation_scores_max_index)           

        return sentence, clusters, mention_loss, coreference_total_loss, relation_loss, mention_scores_max_index, coreference_total_scores_max_index, relation_scores_max_index

    def _extract_mentions(self, sentence, scores):
        mentions = []
        ref = torch.cuda.LongTensor()
        if (self.mode) == 'prediction':
            ref = scores
        else:
            ref = sentence.mention_goldens

        for i in range(len(sentence.words)):
            # O tag
            if (ref[i] == 0):
                if (all(mention.head > -1 for mention in mentions)):
                    self._extract_attention(mentions)
                    sentence.mentions.extend(mentions)
                    mentions.clear()
            # H tag
            elif (ref[i] == 1):
                if (len(mentions) == 0):
                    mention = Mention(sentence.words[i], 0)
                    mentions.append(mention)
                else:
                    if (all(mention.head == -1 for mention in mentions)):
                        for mention in mentions:
                            mention.words.append(sentence.words[i])
                            mention.head = len(mention.words) - 1
                    else:
                        self._extract_attention(mentions)
                        sentence.mentions.extend(mentions)
                        mentions.clear()
                        mention = Mention(sentence.words[i], 0)
                        mentions.append(mention)

            # R tag
            elif (ref[i] == 2):
                if (len(mentions) == 0):
                    mention = Mention(sentence.words[i])
                    mentions.append(mention)
                else:
                    inserted = False
                    for mention in mentions:
                        if (mention.words[-1].position == sentence.words[i].position - 1):
                            mention.words.append(sentence.words[i])
                            inserted = True
                            break 
                    if (not inserted):
                        mention = Mention(sentence.words[i])
                        mentions.append(mention)

    def _extract_clusters(self, sentence, clusters, index, score):
        ref = torch.cuda.LongTensor()
        if (self.mode) == 'prediction':
            ref = score.item()
        else:
            ref = sentence.coreference_goldens[index].item()

        if (ref == 0):
            cluster = Cluster(sentence.mentions[index])
            clusters.append(cluster)
            sentence.mentions[index].cluster = len(clusters) - 1
        else:
            clusters[ref - 1].mentions.append(sentence.mentions[index])
            sentence.mentions[index].cluster = ref - 1

    # Attention is the encoding of the mention's head
    def _extract_attention(self, mentions):
        for mention in mentions:
            if self.technique == 'head':
                mention.attention = mention.words[mention.head].encoding
            elif self.technique == 'padding':
                att = torch.cuda.FloatTensor()
                max = 7*self.encoding_hidden_dim
                for word in mention.words:
                    att = torch.cat((att, word.encoding), dim=1)
                mention.attention = F.pad(att, (0, max-att.size(1)), mode='constant', value=0)
            elif self.technique == 'weightedaverage':
                encodings = mention.words[mention.head].encoding
                weights = torch.cuda.FloatTensor([[len(mention.words)]])
                for i in range(len(mention.words)):
                    if i != mention.head:
                        encodings = torch.cat((encodings, mention.words[i].encoding), dim=0)
                        weights = torch.cat((weights, torch.cuda.FloatTensor([[1]])), dim=0)                    
                weights = F.softmax(weights, dim=0)
                mention.attention = torch.sum(torch.mul(encodings, weights), dim=0).view(1, -1)
            else :
                encodings = torch.cuda.FloatTensor()
                for word in mention.words:
                    encodings = torch.cat((encodings, word.encoding), dim=0)
                alphas = self.attention(encodings)
                weights = F.softmax(alphas, dim=0)
                mention.attention = torch.sum(torch.mul(encodings, weights), dim=0).view(1, -1)
    
    def _extract_relations(self, sentence, clusters, scores):
        ref = torch.cuda.LongTensor()
        if (self.mode) == 'prediction':
            ref = scores
        else:
            ref = sentence.relation_goldens

        reference = (-1 + sqrt(1 + 8 * len(ref))) / 2
        iterator = 0
        for i in range(int(reference) + 1):
            for j in range(i):
                if (ref[iterator] > 1):
                    clusters[sentence.mentions[i].cluster].relations[ref[iterator].item()].append(sentence.mentions[j].cluster)
                iterator += 1


class Training():
    def __init__(self, model, train_corpus, val_corpus, test_corpus, freezed_layers, lr):
        self.train_corpus = train_corpus
        self.val_corpus = val_corpus
        self.test_corpus = test_corpus
        self.freezed_layers = freezed_layers
        self.early_stopping = False
        self.epoch_counter_no_improve = 0
        self.min_loss = 10000
        self.model = cudaize(model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def freeze_model(self):
        if self.model.method == 'BERT':
            counter = 0
            bert_embeddings_layers = [self.model.encodingLayer.bert.embeddings]
            bert_encoder_layers = self.model.encodingLayer.bert.encoder.layer
            bert_pooler_layers = [self.model.encodingLayer.bert.pooler]

            for layer in bert_embeddings_layers:
                if counter < self.freezed_layers:
                    counter += 1
                    for param in layer.parameters():
                        param.requires_grad = False

            for layer in bert_encoder_layers:
                if counter < self.freezed_layers:
                    counter += 1
                    for param in layer.parameters():
                        param.requires_grad = False
                        
            for layer in bert_pooler_layers:
                if counter < self.freezed_layers:
                    counter += 1
                    for param in layer.parameters():
                        param.requires_grad = False
        elif self.model.method == 'GPT':
            counter = 0
            gpt_wte_layers = [self.model.encodingLayer.gpt.wte]
            gpt_wpe_layers = [self.model.encodingLayer.gpt.wpe]
            gpt_drop_layers = [self.model.encodingLayer.gpt.drop]
            gpt_h_layers = self.model.encodingLayer.gpt.h
            gpt_lnf_layers = [self.model.encodingLayer.gpt.ln_f]

            for layer in gpt_wte_layers:
                if counter < self.freezed_layers:
                    counter += 1
                    for param in layer.parameters():
                        param.requires_grad = False

            for layer in gpt_wpe_layers:
                if counter < self.freezed_layers:
                    counter += 1
                    for param in layer.parameters():
                        param.requires_grad = False
                        
            for layer in gpt_drop_layers:
                if counter < self.freezed_layers:
                    counter += 1
                    for param in layer.parameters():
                        param.requires_grad = False

            for layer in gpt_h_layers:
                if counter < self.freezed_layers:
                    counter += 1
                    for param in layer.parameters():
                        param.requires_grad = False
                        
            for layer in gpt_lnf_layers:
                if counter < self.freezed_layers:
                    counter += 1
                    for param in layer.parameters():
                        param.requires_grad = False
        else :
            return

    def train(self, num_epochs):
        self.freeze_model()

        for epoch in range(1, num_epochs + 1): 
            print('EPOCH %d' % epoch)           
            self.model.zero_grad()  
            self.train_epoch()
            self.evaluate_epoch(self.val_corpus, epoch)

            if self.early_stopping == True:
                break

        print('----- TEST -----\n')
        self.evaluate(self.test_corpus)
        print('\n')

    def train_epoch(self):
        self.model.train()

        epoch_mention_loss = torch.Tensor()
        epoch_coreference_loss = torch.Tensor()
        epoch_relation_loss = torch.Tensor()

        for document in self.train_corpus.documents:
            document_mention_loss, document_coreference_loss, document_relation_loss = self.train_document(document)

            epoch_mention_loss = torch.cat((epoch_mention_loss, torch.Tensor([document_mention_loss])), dim=0)
            epoch_coreference_loss = torch.cat((epoch_coreference_loss, torch.Tensor([document_coreference_loss])), dim=0)
            epoch_relation_loss = torch.cat((epoch_relation_loss, torch.Tensor([document_relation_loss])), dim=0)

        print('Training   -- Loss Mention: %f | Loss Coreference: %f | Loss Relation: %f' \
            % (torch.mean(epoch_mention_loss, dim=0), torch.mean(epoch_coreference_loss, dim=0), torch.mean(epoch_relation_loss, dim=0)))

    def train_document(self, document):
        document_mention_loss = 0
        document_coreference_loss = 0
        document_relation_loss = 0

        for i in range(len(document.sentences)):
            self.optimizer.zero_grad()

            _, _, mention_loss, coreference_loss, relation_loss, _, _, _ = self.model(document.sentences[i], document.clusters)

            document_mention_loss += mention_loss.item()
            if coreference_loss != 0:
                document_coreference_loss += coreference_loss.item() 
            if relation_loss != 0:
                document_relation_loss += relation_loss.item() 

            self.optimizer.step()

            document.sentences[i].mentions.clear()

        document.clusters.clear()

        return document_mention_loss, document_coreference_loss, document_relation_loss

    def evaluate_epoch(self, corpus, epoch):
        self.model.eval()

        corpus_mention_loss = torch.Tensor()
        corpus_coreference_loss = torch.Tensor()
        corpus_relation_loss = torch.Tensor()
        corpus_mention_predictions = []
        corpus_coreference_predictions = []
        corpus_relation_predictions = []
        corpus_mention_goldens = []
        corpus_coreference_goldens = []
        corpus_relation_goldens = []

        for document in corpus.documents:
            document_mention_loss, document_coreference_loss, document_relation_loss, document_mention_predictions, document_coreference_predictions, document_relation_predictions, document_mention_goldens, document_coreference_goldens, document_relation_goldens = self.evaluate_document(document)

            corpus_mention_loss = torch.cat((corpus_mention_loss, torch.Tensor([document_mention_loss])), dim=0)
            corpus_coreference_loss = torch.cat((corpus_coreference_loss, torch.Tensor([document_coreference_loss])), dim=0)
            corpus_relation_loss = torch.cat((corpus_relation_loss, torch.Tensor([document_relation_loss])), dim=0)

            corpus_mention_predictions.extend(document_mention_predictions)
            corpus_coreference_predictions.extend(document_coreference_predictions)
            corpus_relation_predictions.extend(document_relation_predictions)

            corpus_mention_goldens.extend(document_mention_goldens)
            corpus_coreference_goldens.extend(document_coreference_goldens)
            corpus_relation_goldens.extend(document_relation_goldens)

        print('Evaluation -- Loss Mention: %f | Loss Coreference: %f | Loss Relation: %f' \
            % (torch.mean(corpus_mention_loss, dim=0), torch.mean(corpus_coreference_loss, dim=0), torch.mean(corpus_relation_loss, dim=0)))

        agregated_loss = torch.mean(corpus_mention_loss, dim=0)*7 + torch.mean(corpus_coreference_loss, dim=0) + torch.mean(corpus_relation_loss, dim=0)*4
        if agregated_loss < self.min_loss:
            self.save_model('modelGptPrediction.pth')
            self.epoch_counter_no_improve = 0
            self.min_loss = agregated_loss 
        else:
            self.epoch_counter_no_improve += 1
        
        if (self.epoch_counter_no_improve > 4) or (epoch > 7 and self.epoch_counter_no_improve > 2):
            self.early_stopping = True

    def evaluate_document(self, document): 
        document_mention_loss = 0
        document_coreference_loss = 0
        document_relation_loss = 0
        document_mention_predictions = []
        document_coreference_predictions = []
        document_relation_predictions = []
        document_mention_goldens = []
        document_coreference_goldens = []
        document_relation_goldens = []

        with torch.no_grad():         
            for i in range(len(document.sentences)):
                _, _, mention_loss, coreference_loss, relation_loss, mention_scores_max_index, coreference_scores_max_index, relation_scores_max_index = self.model(document.sentences[i], document.clusters)

                document_mention_loss += mention_loss.item()
                document_mention_predictions.extend(mention_scores_max_index.tolist())
                document_mention_goldens.extend(document.sentences[i].mention_goldens.tolist())

                if coreference_loss != 0:
                    document_coreference_loss += coreference_loss.item() 
                    document_coreference_predictions.extend(coreference_scores_max_index.tolist())
                    document_coreference_goldens.extend(document.sentences[i].coreference_goldens.tolist())
                if relation_loss != 0:
                    document_relation_loss += relation_loss.item()
                    document_relation_predictions.extend(relation_scores_max_index.tolist())
                    document_relation_goldens.extend(document.sentences[i].relation_goldens.tolist())

                document.sentences[i].mentions.clear()

            document.clusters.clear()

        return document_mention_loss, document_coreference_loss, document_relation_loss, document_mention_predictions, document_coreference_predictions, document_relation_predictions, document_mention_goldens, document_coreference_goldens, document_relation_goldens

    def evaluate(self, corpus):
        self.load_model('modelGptPrediction.pth')
        self.model.eval()

        corpus_mention_loss = torch.Tensor()
        corpus_coreference_loss = torch.Tensor()
        corpus_relation_loss = torch.Tensor()
        corpus_mention_predictions = []
        corpus_coreference_predictions = []
        corpus_relation_predictions = []
        corpus_mention_goldens = []
        corpus_coreference_goldens = []
        corpus_relation_goldens = []

        for document in corpus.documents:
            document_mention_loss, document_coreference_loss, document_relation_loss, document_mention_predictions, document_coreference_predictions, document_relation_predictions, document_mention_goldens, document_coreference_goldens, document_relation_goldens = self.evaluate_document(document)

            corpus_mention_loss = torch.cat((corpus_mention_loss, torch.Tensor([document_mention_loss])), dim=0)
            corpus_coreference_loss = torch.cat((corpus_coreference_loss, torch.Tensor([document_coreference_loss])), dim=0)
            corpus_relation_loss = torch.cat((corpus_relation_loss, torch.Tensor([document_relation_loss])), dim=0)

            corpus_mention_predictions.extend(document_mention_predictions)
            corpus_coreference_predictions.extend(document_coreference_predictions)
            corpus_relation_predictions.extend(document_relation_predictions)

            corpus_mention_goldens.extend(document_mention_goldens)
            corpus_coreference_goldens.extend(document_coreference_goldens)
            corpus_relation_goldens.extend(document_relation_goldens)

        print('--Mentions statistics--\n')
        # mention_report = flat_classification_report(y_pred=corpus_mention_predictions, y_true=corpus_mention_goldens)
        mention_report = classification_report(y_pred=corpus_mention_predictions, y_true=corpus_mention_goldens, digits=3, zero_division=0)
        print(mention_report)

        print('--Coreferences statistics--\n')
        prediction_clusters, golden_clusters = self._extract_clusters(corpus_coreference_predictions, corpus_coreference_goldens)
        precision = bcubed.precision(prediction_clusters, golden_clusters)
        recall = bcubed.recall(prediction_clusters, golden_clusters)
        fscore = bcubed.fscore(precision, recall)
        print('Precision: %f' % precision)
        print('Recall: %f' % recall)
        print('F1: %f \n' % fscore)

        print('--Relations statistics--\n')
        # relation_report = flat_classification_report(y_pred=corpus_relation_predictions, y_true=corpus_relation_goldens)
        relation_report = classification_report(y_pred=corpus_relation_predictions, y_true=corpus_relation_goldens, digits=3, zero_division=0)
        print(relation_report)

    def _extract_clusters(self, predictions, goldens):
        """ Extract clusters from raw corefernce scoring data """
        prediction_clusters = self._extract_clusters_from_list(predictions)
        golden_clusters = self._extract_clusters_from_list(goldens)

        return prediction_clusters, golden_clusters

    def _extract_clusters_from_list(self, list):
        """ Extract clusters from a list """
        clusters = {}
        iterator = 1
        for i in range(len(list)):
            if (list[i] == 0):
                clusters[i + 1] = set([iterator])
                iterator += 1
            else:
                clusters[i + 1] = set(list[i])

        return clusters

    def save_model(self, savepath):
        torch.save(self.model.state_dict(), savepath)

    def load_model(self, loadpath):
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
        self.model = cudaize(self.model)

class Prediction:
    def __init__(self, model):
        self.model = model
        self.load_model('model'+self.model.method + 'Prediction.pth')

    def load_model(self, loadpath):
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
        self.model = cudaize(self.model)

    def predict(self, document):
        self.model.eval()
        
        with torch.no_grad():         
            for i in range(len(document.sentences)):
                document.sentences[i], document.clusters, _, _, _, _, _, _ = self.model(document.sentences[i], document.clusters, 'prediction')
                
        return document


# 'RNN', 'BERT' ou 'GPT'
METHOD = 'GPT'
# 'head', 'padding', 'weightedaverage' ou 'autoattention'
TECHNIQUE = 'weightedaverage'

MENTION_LOSS_FUNCTION = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([1.0, 2.0, 4.0]))
COREFERENCE_LOSS_FUNCTION = nn.CrossEntropyLoss()
RELATION_LOSS_FUNCTION = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([1.0, 4.0, 2.0, 4.0, 2.0]))

EMBEDDING_DIM = 50
# 100 pour RNN, 768 pour BERT ou GPT
ENCODING_HIDDEN_DIM = 768
MENTION_DIM = 3
RELATION_DIM = 5

FREEZED_LAYERS = 12
DECODING_HIDDEN_DIM = 100
DISTANCE_DIM = 50
DROPOUT = 0.2
LEARNING_RATE = 0.001

EPOCHS = 30

embeddings_index, mentions_index = prepare_index()
train_corpus, val_corpus, test_corpus = prepare_corpora(embeddings_index, mentions_index)

model = Model(METHOD, TECHNIQUE, MENTION_LOSS_FUNCTION, COREFERENCE_LOSS_FUNCTION, RELATION_LOSS_FUNCTION, EMBEDDING_DIM, ENCODING_HIDDEN_DIM, DECODING_HIDDEN_DIM, MENTION_DIM, RELATION_DIM, DISTANCE_DIM, DROPOUT)

######################
# Training
training = Training(model, train_corpus, val_corpus, test_corpus, FREEZED_LAYERS, lr=LEARNING_RATE)
training.train(EPOCHS)

######################
# Prediction
prediction = Prediction(model)
resultat = prediction.predict(test_corpus.documents[1])
print()
