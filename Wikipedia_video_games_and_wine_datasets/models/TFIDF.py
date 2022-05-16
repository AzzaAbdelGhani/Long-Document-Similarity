from data.Load_dataset import WikipediaLongDocumentSimilarityDataset
from models.Summarization import Summarization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
import ast
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_ = set(stopwords.words('english'))
import re
import spacy
#nlp = spacy.load('en_core_web_sm')
from spacy.lang.en import English


class TF_IDF:

  def __init__(self,dataset_name=None, use_summarization = False):
    if use_summarization == False:
      self.dataset = WikipediaLongDocumentSimilarityDataset(dataset_name)
      self.sections = [self.preprocessing(section) for section in self.get_sections()]
    else:
      self.sum = Summarization(dataset_name)
      self.dataset = self.sum.dataset
      self.sections = [self.preprocessing(section) for section in self.sum.summaries.values()]
    
    self.tfidf_vectorizer = TfidfVectorizer(analyzer='word',
                                            min_df=0.001,
                                            max_df=0.75,
                                            stop_words='english',
                                            sublinear_tf=True)
    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.sections)
    #print(self.tfidf_matrix.shape)
    
    self.results = self.test_model(k = len(self.dataset))

  def get_sections(self):
    articles_sections = []
    for article in self.dataset.articles:
      sections = ast.literal_eval(article[1]) 
      sentences = []
      for section in sections:
        sentences.append(section[0]) 
        sentences.extend(sent_tokenize(section[1])) 
      articles_sections.append(' '.join([str(item).lower() for item in sentences if item not in stopwords_]))
    return articles_sections

  def preprocessing(self,text):
    #lemmatizer = WordNetLemmatizer()
    #porter = PorterStemmer()
    punctiation_pattern = re.compile('[!-_@#$%^&*()?<>;\.,:"]')
    text = re.sub(punctiation_pattern, '', text)
    numbers_patterns = re.compile('[0-9]+[\w]*')
    text = re.sub(numbers_patterns, '', text)
    #sec = ' '.join([str(lemmatizer.lemmatize(w)).lower() for w in text.split()])
    #sec = ' '.join([str(porter.stem(w)).lower() for w in text.split()])
    #sec = ' '.join([str(w).lower() for w in text.split()])
    return text

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
    for doc, labels in tqdm(self.dataset.labels.items(), desc="Find k = {} Similar articles".format(k)):
      model_labels[doc] = self.get_cosine_similarities(doc, len(labels.keys()))
    return model_labels