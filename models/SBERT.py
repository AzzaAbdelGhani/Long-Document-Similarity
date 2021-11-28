from sentence_transformers import SentenceTransformer
from data import Load_dataset
import ast
import torch 

class SBERT:

  def __init__(self,dataset_name):
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    self.dataset = Load_dataset.WikipediaLongDocumentSimilarityDataset(dataset_name)
    self.articles_embeddings = self.compute_articles_embeddings()

  def compute_articles_embeddings(self):
    embeddings = []
    for article in self.dataset.articles[:2]: 
      sections = ast.literal_eval(article[1]) #extract the sections
      sentences = []
      for section in sections:
        sentences.append(section[0]) #store section's title 
        sentences.append(section[1]) #stroe section's description 
      embeddings_list = self.model.encode(sentences, convert_to_tensor=True)
      embeddings.append(torch.mean(embeddings_list, dim=0)) #average embeddings of all sentences
    return embeddings
