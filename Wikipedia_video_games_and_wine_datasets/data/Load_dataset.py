import os 
from torch.utils.data import Dataset
import csv
import sys
import pandas as pd

class WikipediaLongDocumentSimilarityDataset(Dataset):
  
  def __init__(self,dataset_name):
    self.raw_data_path = self.download_raw(dataset_name)
    self.articles = self.read_all_articles()
    self.titles = [article[0] for article in self.articles]
    self.labels = self.read_ground_truth_labels(dataset_name)
    
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

  def read_ground_truth_labels(self, dataset_name):
    ground_truth_path = f"data/Ground-Truth/{dataset_name}/gt"
    labeled_data = pd.read_pickle(ground_truth_path)
    for doc in list(labeled_data):
      if doc not in self.titles:
        labeled_data.pop(doc)
        continue
      for label in list(labeled_data[doc]):
        if label not in self.titles:
          labeled_data[doc].pop(label)
    return labeled_data
        
  def __len__(self):
    return(len(self.articles))