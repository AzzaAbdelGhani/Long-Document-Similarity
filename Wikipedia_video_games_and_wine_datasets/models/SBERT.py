from sentence_transformers import SentenceTransformer, util
from data import Load_dataset
import ast
import torch 
import faiss
import numpy
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import pandas as pd
from tqdm import tqdm
import io
import pickle

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


class SBERT:

  def __init__(self,dataset_name, saved_embeddings = None, device = None, save = False):
    self.dataset = Load_dataset.WikipediaLongDocumentSimilarityDataset(dataset_name)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu" # establish device
    #print(device)

    self.titles = [article[0] for article in self.dataset.articles]
    
    if saved_embeddings == None:
      self.model = SentenceTransformer('all-roberta-large-v1', device=device)
      self.model.max_seq_length = self.model.max_seq_length 
      self.articles_embeddings = self.compute_articles_embeddings()
      if save == True: #Stroe embeddings
        with open(dataset_name+'_embeddings.pkl', "wb") as fOut:
          pickle.dump(self.articles_embeddings, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    else:
      if device == "cpu":
        with open(saved_embeddings, "rb") as f:
          self.articles_embeddings = CPU_Unpickler(f).load()
      else:
        self.articles_embeddings = pd.read_pickle(saved_embeddings) 
    self.results = self.test_model(k = len(self.dataset)-1)  

  def split_sentence(self,sentence):
    if len(sentence.split()) > 200:
        s = []
        s.append(' '.join(sentence.split()[:200]))
        s.append(' '.join(self.split_sentence(' '.join(sentence.split()[200:]))))
        return s
    else: 
        return [sentence]

  def compute_articles_embeddings(self):
    embeddings = []
    for article in tqdm(self.dataset.articles, desc="Compute Embeddings for each document"): 
      sections = ast.literal_eval(article[1]) #extract article's sections
      sentences = []
      for section in sections:
        sentences.append(section[0]) #store section's title 
        for sent in sent_tokenize(section[1]):
            s = self.split_sentence(sent)
            sentences.extend(s)
        
      embeddings_list = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
      embeddings.append(torch.mean(embeddings_list, dim=0)) #average embeddings of all sentences
    return embeddings

  def compute_score_similarity(self,video_game_1_title, video_game_2_title):
    idx1 = self.titles.index(video_game_1_title)
    idx2 = self.titles.index(video_game_2_title)
    embed1 = self.articles_embeddings[idx1]
    embed2 = self.articles_embeddings[idx2]
    cosine_score = util.pytorch_cos_sim(embed1, embed2)
    return cosine_score.item()

  def faiss_index(self,query_idx, query, k):
    d = len(self.articles_embeddings[0]) #embedding's size 
    n = len(self.articles_embeddings) #number of articles
    other_articles_embeddings = self.articles_embeddings[:query_idx] + self.articles_embeddings[query_idx+1:]
    document_embeddings = numpy.array([numpy.array(x.cpu()) for x in other_articles_embeddings])
    index = faiss.IndexFlatL2(d)   # build the index, d=size of vectors 
    index.add(document_embeddings)                  
    D, I = index.search(query, k) 
    I_titles = [] 
    for i in I[0]:
      I_titles.append(self.titles[i])
    return(I_titles)

  def find_similar_docs(self, video_game_title, num_items):
    if video_game_title not in self.titles:
      return []
    idx = self.titles.index(video_game_title)
    query_embed = self.articles_embeddings[idx].cpu().numpy().reshape(1,len(self.articles_embeddings[idx]))
    similar_docs = self.faiss_index(idx, query=query_embed, k=num_items)
    return similar_docs

  def test_model(self, k=10):
    model_labels = {}
    for doc in tqdm(self.dataset.labels.keys(), desc="Find k = {} Similar articles".format(k)):
      model_labels[doc] = self.find_similar_docs(doc, k)
    return model_labels
    
