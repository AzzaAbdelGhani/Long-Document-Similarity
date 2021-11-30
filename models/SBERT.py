from sentence_transformers import SentenceTransformer, util
from data import Load_dataset
import ast
import torch 
import faiss
import numpy

class SBERT:

  def __init__(self,dataset_name):
    self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    self.dataset = WikipediaLongDocumentSimilarityDataset(dataset_name)
    self.articles_embeddings = self.compute_articles_embeddings()
    self.titles = [article[0] for article in self.dataset.articles]
    

  def compute_articles_embeddings(self):
    embeddings = []
    for article in self.dataset.articles: 
      sections = ast.literal_eval(article[1]) #extract the sections
      sentences = []
      for section in sections:
        sentences.append(section[0]) #store section's title 
        sentences.append(section[1]) #stroe section's description 
      embeddings_list = self.model.encode(sentences, convert_to_tensor=True)
      embeddings.append(torch.mean(embeddings_list, dim=0)) #average embeddings of all sentences
    return embeddings

  def compute_score_similarity(self,video_game_1_title, video_game_2_title):
    idx1 = self.titles.index(video_game_1_title)
    idx2 = self.titles.index(video_game_2_title)
    #print(idx1)
    #print(idx2)
    embed1 = self.articles_embeddings[idx1]
    embed2 = self.articles_embeddings[idx2]
    cosine_score = util.pytorch_cos_sim(embed1, embed2)
    return cosine_score.item()

  def faiss_index(self, query, k):
    d = len(self.articles_embeddings[0]) #embedding's size 
    n = len(self.articles_embeddings) #number of articles
    document_embeddings = numpy.array([numpy.array(x.cpu()) for x in self.articles_embeddings])
    index = faiss.IndexFlatL2(d)   # build the index, d=size of vectors 
    index.add(document_embeddings)                  
    #print(index.ntotal)
    # we want 4 similar vectors
    D, I = index.search(query, k)     # actual search
    for i in I[0]:
      print(self.titles[i])

  def find_similar_docs(self, video_game_title, num_items):
    idx = self.titles.index(video_game_title)
    query_embed = self.articles_embeddings[idx].cpu().numpy().reshape(1,len(self.articles_embeddings[idx]))
    similar_docs = self.faiss_index(query=query_embed, k=num_items)
    return similar_docs
