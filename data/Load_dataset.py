import os 
from torch.utils.data import Dataset
import csv
import sys

class WikipediaLongDocumentSimilarityDataset(Dataset):
  
  def __init__(self,dataset_name):
    self.raw_data_path = self.download_raw(dataset_name)
    self.articles = self.read_all_articles()
    
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
    
  def __len__(self):
    return(len(self.articles))


