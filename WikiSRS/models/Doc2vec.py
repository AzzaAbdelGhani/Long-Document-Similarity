import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm
import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from spacy.lang.ar.stop_words import STOP_WORDS as ar_stop
from spacy.lang.es.stop_words import STOP_WORDS as es_stop
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.ru.stop_words import STOP_WORDS as ru_stop
from spacy.lang.pt.stop_words import STOP_WORDS as pt_stop
from tashaphyne.stemming import ArabicLightStemmer
from nltk.stem import SnowballStemmer
from scipy import stats 
from scipy.spatial import distance
#from sklearn.metrics.pairwise import cosine_similarity


class Doc2vec_WikiSimilarity():
	def __init__(self, dataset_name=None, clean=False, train=False, test=False):
		self.dataset_name = dataset_name
		self.wiki_languages = {'en':'english','ar':'arabic','es':'spanish','fr':'french','ru':'russian','pt':'portuguese'}
		self.multilingual_data, self.data = self.load_dataset(dataset_name,clean)
		if test == True:
			self.en_model = Doc2Vec.load("data/doc2vec_models/en_d2v.model")
			self.ar_model = Doc2Vec.load("data/doc2vec_models/ar_d2v.model")
			self.es_model = Doc2Vec.load("data/doc2vec_models/es_d2v.model")
			self.fr_model = Doc2Vec.load("data/doc2vec_models/fr_d2v.model")
			self.pt_model = Doc2Vec.load("data/doc2vec_models/pt_d2v.model")
			self.ru_model = Doc2Vec.load("data/doc2vec_models/ru_d2v.model")
			self.inferred_vectors = self.get_inferred_vectors()
		self.similarity_scores= self.get_similarity_scores()
		self.ranking_accuracy, self.r_s = self.evaluate_results() 

	def preprocessing(self,language,content):
		multilingual_stopwords = list(en_stop)+list(ar_stop)+list(es_stop)+list(fr_stop)+list(ru_stop)+list(pt_stop)
		text = ' '.join([c.lower() for c in content.split() if c not in multilingual_stopwords])
		punctiation_pattern = re.compile('[!-_@#$%^&*()?<>;\.,:"]')
		text = re.sub(punctiation_pattern, '', text)
		numbers_patterns = re.compile('[0-9]+[\w]*')
		text = re.sub(numbers_patterns, '', text)
		clean_text = text
		if language == 'ar':
			ArListem = ArabicLightStemmer()
			clean_text = ' '.join([ArListem.light_stem(token) for token in text.split(" ")])
		else :
			stemmer = SnowballStemmer(self.wiki_languages[language])
			clean_text = ' '.join([stemmer.stem(token) for token in text.split(" ")])

		return clean_text       
		
	def cleaning_data(self, data):
		clean_content = []
		for lang, content in tqdm(zip(list(data.lang), list(data.content)), desc="Cleaning dataset "):
			clean_content.append(self.preprocessing(lang,content))
		return clean_content
        
        
	def load_dataset(self, dataset_name, clean):
		if dataset_name == "wikipediaSimilarity353":
		    data = pd.read_csv("data/wikipediaSimilarity353.csv")
		    data['titleA'] = data['titleA'].replace(['Production, costs, and pricing'],'Production')#no wikipedia page for 'Production, costs, and pricing'
		elif dataset_name == "WikiSRS_relatedness" or dataset_name == "WikiSRS_similarity":
		    data = pd.read_csv("data/"+dataset_name+".csv", sep='\t') 
		    data = data.drop(['RawScores', 'StdDev'], axis = 1)
		    data.rename(columns = {'Term1':'termA', 'String1':'titleA',
		                           'Term2':'termB', 'String2':'titleB',
		                           'Mean' :'relatedness'}, inplace = True)
		    data['relatedness'] = data['relatedness'].div(10)
		multilingual_data = pd.read_csv("data/multilingual_"+dataset_name+".csv") 
		#multilingual_data = multilingual_data[:100]
		if clean == True:
			multilingual_data['clean_content'] = self.cleaning_data(multilingual_data)
		return multilingual_data, data

	def get_inferred_vectors(self):
		inferred_vectors = {}
		inferred_vectors['en'] = {}
		inferred_vectors['ar'] = {}
		inferred_vectors['es'] = {}
		inferred_vectors['fr'] = {}
		inferred_vectors['ru'] = {}
		inferred_vectors['pt'] = {}

		for title,lang,clean_content in tqdm(zip(list(self.multilingual_data.title),list(self.multilingual_data.lang), list(self.multilingual_data.clean_content)),desc="Inferring vectors"):
		    if lang == 'en':
		    	doc2vec_model = self.en_model
		    elif lang == 'ar':
		        doc2vec_model = self.ar_model
		    elif lang == 'es':
		        doc2vec_model= self.es_model
		    elif lang == 'fr':
		        doc2vec_model = self.fr_model
		    elif lang == 'ru':
		        doc2vec_model = self.ru_model
		    elif lang == 'pt':
		        doc2vec_model = self.pt_model

		    inferred_vectors[lang][title] = doc2vec_model.infer_vector(word_tokenize(clean_content))
		
		return inferred_vectors

	def save_similarity_scores(self, results):
		with open("results/"+self.dataset_name+"/doc2vec_similarity_scores.txt", "w") as f:
			for key, value in results.items():
				#f.write('%s: %s\n' % (key,value))
				f.write('%s :\n' % key)
				for item in value: 
					f.write('%s\n' % item)
		return

	def get_similarity_scores(self): 
		results = {}
		results['en'] = []
		results['ar'] = []
		results['es'] = []
		results['fr'] = []
		results['ru'] = []
		results['pt'] = []
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
		for idx, row in tqdm(self.data.iterrows(), desc="Find Similar Articles "): 
			for lang in list(self.wiki_languages.keys()):
				if row['titleA'] in list(self.inferred_vectors[lang].keys()) and row['titleB'] in list(self.inferred_vectors[lang].keys()):
					embedA = self.inferred_vectors[lang][row['titleA']]
					embedB = self.inferred_vectors[lang][row['titleB']]

			    #Think about using two metrics : spatial.cosine.distance() and spatial.distance.correlation()
					sim_score = 1 - distance.cosine(embedA, embedB) #1-distance.cosine returns the similarity 
					#sim_score = cosine_similarity([embedA], [embedB]) # this method returns the same results of distance.cosine 
					results[lang].append({'titleA':row['titleA'], 'titleB':row['titleB'], 'predicted':float(sim_score), 'actual':row['relatedness']/10.0})
		self.save_similarity_scores(results)
		return results

	def save_metrics_results(self, Average_of_true_ranks, r_s):
		with open("results/"+self.dataset_name+"/doc2vec_metrics_results", "w") as f :
			f.write("Average of True Rankings for each language : \n")
			for k, v in Average_of_true_ranks.items():
				f.write('%s: %s\n' % (k, v))
			f.write("\nSpearman Correlation Results for each language : \n")
			for k, v in r_s.items():
				f.write('%s: %s\n' % (k, v))
		return

	def evaluate_results(self): 
		true_rank = {'en':0,'ar':0,'es':0,'fr':0,'ru':0,'pt':0}
		Average_of_true_ranks = {'en':0,'ar':0,'es':0,'fr':0,'ru':0,'pt':0}
		counter = {'en':0,'ar':0,'es':0,'fr':0,'ru':0,'pt':0}
		r_s = {'en':0,'ar':0,'es':0,'fr':0,'ru':0,'pt':0}

		for lang in self.wiki_languages.keys(): 
			lang_results = self.similarity_scores[lang]
			for rowA in lang_results: 
				for rowB in lang_results:
					if rowA['titleA'] == rowB['titleA'] and rowA['titleB'] == rowB['titleB'] : 
						continue
					counter[lang] += 1
					if rowA['actual'] < rowB['actual'] and rowA['predicted'] < rowB['predicted']:
						true_rank[lang] += 1
					elif rowA['actual'] > rowB['actual'] and rowA['predicted'] > rowB['predicted']:
						true_rank[lang] +=1 
			Average_of_true_ranks[lang] = true_rank[lang]/counter[lang]
			r_s[lang], p_value = stats.spearmanr(pd.DataFrame(lang_results).actual, pd.DataFrame(lang_results).predicted)
		self.save_metrics_results(Average_of_true_ranks, r_s)
		return Average_of_true_ranks, r_s