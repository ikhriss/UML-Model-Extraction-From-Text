import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
import utils
from model.data_loader import Mention, Cluster
from math import sqrt

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
        self.bert = utils.cudaize(BertModel.from_pretrained('bert-base-uncased'))
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence):
        word_sentence = [word.wording for word in sentence]
        inputs = self.bert_tokenizer.encode(word_sentence)
        outputs = self.bert(input_ids=utils.cudaize(torch.tensor(inputs).view(-1, len(inputs))), attention_mask=utils.cudaize(torch.ones(1, len(inputs), dtype=torch.long)), token_type_ids=utils.cudaize(torch.zeros(1, len(inputs), dtype=torch.long)))
        return self.dropout(outputs.last_hidden_state[0][1:-1])


class EncodingGPTLayer(nn.Module):
    # GPT
    def __init__(self, dropout):
        super().__init__()
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt = utils.cudaize(GPT2Model.from_pretrained('gpt2'))
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence):
        word_sentence = [word.wording for word in sentence]
        inputs = self.gpt_tokenizer.encode(word_sentence)
        outputs = self.gpt(input_ids=utils.cudaize(torch.tensor(inputs).view(-1, len(inputs))), attention_mask=utils.cudaize(torch.ones(1, len(inputs), dtype=torch.long)), token_type_ids=utils.cudaize(torch.zeros(1, len(inputs), dtype=torch.long)))
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

    
class Net(nn.Module):
    def __init__(self, method, technique, mention_loss_function, coreference_loss_function, relation_loss_function, params):
        super().__init__()
        self.mode = 'training'
        self.method = method
        self.technique = technique
        self.mention_loss_function = mention_loss_function
        self.coreference_loss_function = coreference_loss_function
        self.relation_loss_function = relation_loss_function
        self.embedding_dim = params.EMBEDDING_DIM
        self.encoding_hidden_dim = params.ENCODING_HIDDEN_DIM
        self.decoding_hidden_dim = params.DECODING_HIDDEN_DIM
        self.mention_dim = params.MENTION_DIM
        self.relation_dim = params.RELATION_DIM
        self.distance_dim = params.DISTANCE_DIM
        self.padding_dim_coef = 1
        if self.technique == 'padding':
            self.padding_dim_coef = 7   # Max size padding  
                      
        if self.method == 'RNN':
            self.encodingLayer = EncodingRNNLayer(self.embedding_dim, self.encoding_hidden_dim, params.DROPOUT)
            self.encoding_hidden_dim = 2*params.ENCODING_HIDDEN_DIM    # Bilateral
        elif self.method == 'BERT':             
            self.encodingLayer = EncodingBERTLayer(params.DROPOUT)
        else:
            self.encodingLayer = EncodingGPTLayer(params.DROPOUT)
        self.mentionScoreLayer = ScoreLayer(self.encoding_hidden_dim, self.decoding_hidden_dim, self.mention_dim, params.DROPOUT)
        self.attentionLayer = ScoreLayer(self.encoding_hidden_dim, self.decoding_hidden_dim, 1, params.DROPOUT)
        self.coreferenceDistanceLayer = DistanceLayer(self.distance_dim, params.DROPOUT)
        self.coreferenceScoreTLayer = ScoreTLayer(self.encoding_hidden_dim*3*self.padding_dim_coef + self.distance_dim, self.decoding_hidden_dim, 1, params.DROPOUT)
        self.relationDistanceLayer = DistanceLayer(self.distance_dim, params.DROPOUT)
        self.relationScoreLayer = ScoreLayer(self.encoding_hidden_dim*3*self.padding_dim_coef + self.distance_dim, self.decoding_hidden_dim, self.relation_dim, params.DROPOUT)

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
                alphas = self.attentionLayer(encodings)
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

