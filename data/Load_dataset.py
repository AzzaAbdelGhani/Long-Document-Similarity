import os
import torch 
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import csv
import sys
import ast

class WikipediaLongDocumentSimilarityDataset(Dataset):
  
  def __init__(self,dataset_name):
  
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    self.raw_data_path = self.download_raw(dataset_name)
    self.articles = self.read_all_articles()
    self.articles_embeddings = self.compute_articles_embeddings()
    
  def raw_data_link(self, dataset_name):
    if dataset_name == "wines":
        return "https://zenodo.org/record/4812960/files/wines.txt?download=1"
    if dataset_name == "video_games":
        return "https://zenodo.org/record/4812962/files/video_games.txt?download=1"

  def download_raw(self, dataset_name):
          raw_data_path = f"data/datasets/{dataset_name}/raw_data"
          os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
          if not os.path.exists(raw_data_path):
              os.system(f"wget -O {raw_data_path} {self.raw_data_link(dataset_name)}")
          return raw_data_path

  def read_all_articles(self):
        csv.field_size_limit(sys.maxsize)
        with open(self.raw_data_path, newline="") as f:
            reader = csv.reader(f)
            all_articles = list(reader)
        return all_articles[1:] #ignore data[0] --> ['Title','Sections']
        
  def compute_articles_embeddings(self):
    embeddings = []
    for article in self.articles[:3]: 
      sections = ast.literal_eval(article[1]) #extract the sections
      sentences = []
      for section in sections:
        sentences.append(section[0]) #store section's title 
        section_body = section[1].split(".")
        for sentence in section_body:
          sentences.append(sentence)
      embeddings_list = self.model.encode(sentences, convert_to_tensor=True)
      embeddings.append(torch.mean(embeddings_list, dim=0)) #average embeddings of all sentences
    return embeddings
      

  def __len__(self):
    return(len(self.articles))


