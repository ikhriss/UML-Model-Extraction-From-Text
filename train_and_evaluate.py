"""Train the model"""

import argparse
import logging
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import bcubed
import utils
import model.net as net
from model.data_loader import prepare_index, prepare_corpora
from predict import Prediction
import torch.nn as nn
from sklearn.metrics import classification_report


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--METHOD', default='BERT',
                    help="RNN, BERT ou GPT")
parser.add_argument('--TECHNIQUE', default='weightedaverage',
                    help="head, padding, weightedaverage ou autoattention")

parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


class Training():
    def __init__(self, model, train_corpus, val_corpus, test_corpus, freezed_layers, lr):
        self.train_corpus = train_corpus
        self.val_corpus = val_corpus
        self.test_corpus = test_corpus
        self.freezed_layers = freezed_layers
        self.early_stopping = False
        self.epoch_counter_no_improve = 0
        self.min_loss = 10000
        self.model = utils.cudaize(model)
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
            logging.info("Epoch {}/{}".format(epoch, params.EPOCHS))           
            self.model.zero_grad()  
            self.train_epoch()
            self.evaluate_epoch(self.val_corpus, epoch)

            if self.early_stopping == True:
                break

        logging.info('----- TEST -----')
        self.evaluate(self.test_corpus)

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

        logging.info('Training   -- Loss Mention: %f | Loss Coreference: %f | Loss Relation: %f' \
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

        logging.info('Evaluation -- Loss Mention: %f | Loss Coreference: %f | Loss Relation: %f' \
            % (torch.mean(corpus_mention_loss, dim=0), torch.mean(corpus_coreference_loss, dim=0), torch.mean(corpus_relation_loss, dim=0)))

        agregated_loss = torch.mean(corpus_mention_loss, dim=0)*7 + torch.mean(corpus_coreference_loss, dim=0) + torch.mean(corpus_relation_loss, dim=0)*4
        if agregated_loss < self.min_loss:
            utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict()},
                              checkpoint=args.model_dir)

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
        utils.load_checkpoint(os.path.join(args.model_dir,'last.pth.tar'), self.model)
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

        logging.info('--Mentions statistics--\n')
        # mention_report = flat_classification_report(y_pred=corpus_mention_predictions, y_true=corpus_mention_goldens)
        mention_report = classification_report(y_pred=corpus_mention_predictions, y_true=corpus_mention_goldens, digits=3, zero_division=0)
        logging.info(mention_report)
        
        logging.info('--Coreferences statistics--\n')
        prediction_clusters, golden_clusters = self._extract_clusters(corpus_coreference_predictions, corpus_coreference_goldens)
        precision = bcubed.precision(prediction_clusters, golden_clusters)
        recall = bcubed.recall(prediction_clusters, golden_clusters)
        fscore = bcubed.fscore(precision, recall)
        logging.info('Precision: %f' % precision)
        logging.info('Recall: %f' % recall)
        logging.info('F1: %f ' % fscore)

        
        logging.info('--Relations statistics--\n')
        # relation_report = flat_classification_report(y_pred=corpus_relation_predictions, y_true=corpus_relation_goldens)
        relation_report = classification_report(y_pred=corpus_relation_predictions, y_true=corpus_relation_goldens, digits=3, zero_division=0)
        logging.info(relation_report)

        
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
        self.model = utils.cudaize(self.model)



if __name__ == '__main__':

  # Load the parameters from json file
  args = parser.parse_args()
  json_path = os.path.join(args.model_dir, 'params.json')
  assert os.path.isfile(
      json_path), "No json configuration file found at {}".format(json_path)
  params = utils.Params(json_path)

  # use GPU if available
  params.cuda = torch.cuda.is_available()

  # Set the random seed for reproducible experiments
  torch.manual_seed(230)
  if params.cuda:
      torch.cuda.manual_seed(230)
  
  MENTION_LOSS_FUNCTION = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 2.0, 4.0]))
  COREFERENCE_LOSS_FUNCTION = nn.CrossEntropyLoss()
  RELATION_LOSS_FUNCTION = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 4.0, 2.0, 4.0, 2.0]))
  
  # Set the logger
  utils.set_logger(os.path.join(args.model_dir, 'train.log'))

  # Create the input data pipeline
  logging.info("Loading the datasets...")

  # Prepare data
  embeddings_index, mentions_index = prepare_index()
  train_corpus, val_corpus, test_corpus = prepare_corpora(embeddings_index, mentions_index)

  logging.info("- done.")

  # Define the model and optimizer
  model = net.Net(args.METHOD, args.TECHNIQUE, MENTION_LOSS_FUNCTION, COREFERENCE_LOSS_FUNCTION, RELATION_LOSS_FUNCTION, params).cuda() if params.cuda else net.Net(args.METHOD, args.TECHNIQUE, MENTION_LOSS_FUNCTION, COREFERENCE_LOSS_FUNCTION, RELATION_LOSS_FUNCTION, params)
    

  # Train the model
  logging.info("Starting training for {} epoch(s)".format(params.EPOCHS))
  training = Training(model, train_corpus, val_corpus, test_corpus, params.FREEZED_LAYERS, lr=params.LEARNING_RATE)
  training.train(params.EPOCHS)