import torch
import utils
import logging
import os
import argparse
import torch.nn as nn
import model.net as net
from model.data_loader import prepare_index, prepare_corpora

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--METHOD', default='BERT',
                    help="RNN, BERT ou GPT")
parser.add_argument('--TECHNIQUE', default='weightedaverage',
                    help="head, padding, weightedaverage ou autoattention")

class Prediction:
    def __init__(self, model, model_dir):
        self.model = model
        utils.load_checkpoint(os.path.join(model_dir,'last.pth.tar'), self.model)

    def load_model(self, loadpath):
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
        self.model = utils.cudaize(self.model)

    def predict(self, document):
        self.model.eval()
        
        with torch.no_grad():         
            for i in range(len(document.sentences)):
                document.sentences[i], document.clusters, _, _, _, _, _, _ = self.model(document.sentences[i], document.clusters, 'prediction')
                
        return document

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

  # Prepare data
  embeddings_index, mentions_index = prepare_index()
  train_corpus, val_corpus, test_corpus = prepare_corpora(embeddings_index, mentions_index)

  # Define the model and optimizer
  model = net.Net(args.METHOD, args.TECHNIQUE, MENTION_LOSS_FUNCTION, COREFERENCE_LOSS_FUNCTION, RELATION_LOSS_FUNCTION, params).cuda() if params.cuda else net.Net(args.METHOD, args.TECHNIQUE, MENTION_LOSS_FUNCTION, COREFERENCE_LOSS_FUNCTION, RELATION_LOSS_FUNCTION, params)
  # Prediction
  prediction = Prediction(model, args.model_dir)
  resultat = prediction.predict(test_corpus.documents[1])
  resultat.save('document_test.txt')
  resultat.show_document()