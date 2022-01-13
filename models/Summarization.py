from data.Load_dataset import WikipediaLongDocumentSimilarityDataset
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stopwords_ = set(stopwords.words('english'))
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import ast
import re
import heapq

class Summarization():
  def __init__(self,dataset_name):
    self.dataset = WikipediaLongDocumentSimilarityDataset(dataset_name)
    self.word_frequencies = self.find_weighted_frequencies()
    self.sentences_scores = self.calculate_sentences_scores()
    self.summaries = self.get_article_summary()

  def find_weighted_frequencies(self):
    articles_words_frequencies = {}
    for article in self.dataset.articles: 
      sections = ""
      word_frequencies = {}
      for section in ast.literal_eval(article[1]):
        #sections += section[0]
        for sent in sent_tokenize(section[1]):
          sections += sent
      sections = re.sub('[!-_@#$%^&*()?<>;\.,:"]', '', sections)
      sections = re.sub('[0-9]+[\w]*', '', sections)
      for word in nltk.word_tokenize(sections):
        if word not in stopwords_ and len(word) > 2:
          if word not in word_frequencies.keys():
            word_frequencies[word] = 1
          else:
            word_frequencies[word] += 1
      maximum_frequncy = max(word_frequencies.values())
      for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
      articles_words_frequencies[article[0]] = word_frequencies     
    return articles_words_frequencies

  def calculate_sentences_scores(self):
    articles_sentences_scores = {}
    for article in self.dataset.articles: 
      sentences = []
      sentence_scores = {}
      for section in ast.literal_eval(article[1]):
        sentences.append(section[0]) #store section's title 
        sentences.extend(sent_tokenize(section[1])) #store section's body
      for sent in sentences:
        for word in nltk.wordpunct_tokenize(sent.lower()):
          if word in self.word_frequencies[article[0]].keys():
            if len(sent.split(' ')) < 30:
              if sent not in sentence_scores.keys():
                sentence_scores[sent] = self.word_frequencies[article[0]][word]
              else:
                sentence_scores[sent] += self.word_frequencies[article[0]][word]
      articles_sentences_scores[article[0]] = sentence_scores
    return articles_sentences_scores

  def get_article_summary(self):
    summaries = {}
    for article in self.dataset.articles:
      summary_sentences = heapq.nlargest(200, self.sentences_scores[article[0]], key=self.sentences_scores[article[0]].get)
      summaries[article[0]] = ' '.join(summary_sentences)
    return summaries