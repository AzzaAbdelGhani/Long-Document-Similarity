from data import Load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
import ast
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

class TF_IDF:

  def __init__(self,dataset_name):
    self.dataset = Load_dataset.WikipediaLongDocumentSimilarityDataset(dataset_name)
    self.sections = self.get_sections()
    self.tfidf_matrix = TfidfVectorizer().fit_transform(self.sections)
    print(self.tfidf_matrix.shape)
    self.results = self.test_model()

  def get_sections(self):
    articles_sections = []
    for article in self.dataset.articles:
      sections = ast.literal_eval(article[1]) 
      sentences = []
      for section in sections:
        sentences.append(section[0]) 
        sentences.extend(sent_tokenize(section[1])) 
      articles_sections.append(' '.join([str(item) for item in sentences]))
    return articles_sections

  def get_cosine_similarities(self, title, k):
    if title not in self.dataset.titles:
      return []
    title_idx = self.dataset.titles.index(title)
    cosine_scores = linear_kernel(self.tfidf_matrix[title_idx], self.tfidf_matrix).flatten()
    similar_articles_indices = cosine_scores.argsort()[:-k:-1]
    similar_titles = [] 
    for i in similar_articles_indices[1:]:
      similar_titles.append(self.dataset.titles[i])
    return(similar_titles)
    
  def test_model(self, k=None):
    model_labels = {}
    for doc, labels in tqdm(self.dataset.labels.items(), desc="Find Similar articles"):
      if k ==  None:
        model_labels[doc] = self.get_cosine_similarities(doc, len(labels.keys())+1)
      else:
        model_labels[doc] = self.get_cosine_similarities(doc, k+1)
    return model_labels