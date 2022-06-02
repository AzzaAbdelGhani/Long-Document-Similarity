import torch
import pandas as pd
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import pickle
import io
import os
import numpy
from sklearn.metrics import accuracy_score
import re
import time
from scipy import stats 


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class SBERT_WikiSimilarity() : 
    def __init__(self, dataset_name=None, save=False, saved_embeddings_path = None):
        self.dataset_name = dataset_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.multilingual_dataset, self.dataset = self.load_dataset(dataset_name)
        self.wiki_languages = ['en','ar','es','fr','ru','pt']
        if saved_embeddings_path == None:
            self.model = SentenceTransformer('distiluse-base-multilingual-cased')
            self.model.max_seq_length = self.model.get_max_seq_length() # For this pre-trained model, max_seq_length is 128
            self.pages_embeddings = self.compute_pages_embeddings()
            if save==True:
              with open('WikiSRS/data/embeddings/'+dataset_name+'_embeddings.pkl', "wb") as fOut:
                pickle.dump(self.pages_embeddings, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            if self.device == "cpu":
                with open(saved_embeddings_path, "rb") as f:
                    self.pages_embeddings = CPU_Unpickler(f).load()
            else:
                self.pages_embeddings = pd.read_pickle(saved_embeddings_path) 

        self.similarity_scores = self.get_similarity_scores()
        self.ranking_accuracy, self.r_s = self.evaluate_ranking() 

    def load_dataset(self, dataset_name):
        multilingual_data = pd.read_csv("WikiSRS/data/multilingual_"+dataset_name+".csv") 
        if dataset_name == "wikipediaSimilarity353":
            data = pd.read_csv("WikiSRS/data/wikipediaSimilarity353.csv")
            data['titleA'] = data['titleA'].replace(['Production, costs, and pricing'],'Production')#no wikipedia page for 'Production, costs, and pricing'
        elif dataset_name == "WikiSRS_relatedness" or dataset_name == "WikiSRS_similarity":
            data = pd.read_csv("WikiSRS/data/"+dataset_name+".csv", sep='\t') 
            data = data.drop(['RawScores', 'StdDev'], axis = 1)
            data.rename(columns = {'Term1':'termA', 'String1':'titleA',
                                   'Term2':'termB', 'String2':'titleB',
                                   'Mean' :'relatedness'}, inplace = True)
            data['relatedness'] = data['relatedness'].div(10)
        return multilingual_data, data
    
    def split_sentence(self,sentence):
        if len(sentence.split()) > 120:
            s = []
            s.append(' '.join(sentence.split()[:120]))
            s.append(' '.join(self.split_sentence(' '.join(sentence.split()[120:]))))
            return s
        else: 
            return [sentence]

    def compute_pages_embeddings(self):
        embeddings = []
        for id, row in tqdm(self.multilingual_dataset.iterrows(), desc="compute page's Embeddings: "):
            sentences = [row.title]
            for sentence in sent_tokenize(row.content):
                s = self.split_sentence(sentence)
                sentences.extend(s)
            embeddings_list = self.model.encode(sentences)
            embeddings.append({"title": row.title, "lang": row.lang, "embedding": torch.mean(self.model.encode(sentences, convert_to_tensor=True), dim=0)}) 
        return embeddings

    def save_similarity_scores(self, results):
        with open("WikiSRS/results/"+self.dataset_name+"/sbert_similarity_scores.txt", "w") as f:
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
        for idx,row in self.dataset.iterrows():
          for lang in self.wiki_languages:
              try :
                  embedA = next(item for item in self.pages_embeddings if item["title"] == row['titleA'] and item['lang'] == lang)['embedding']
                  embedB = next(item for item in self.pages_embeddings if item["title"] == row['titleB'] and item['lang'] == lang)['embedding']
              except:
                  continue
              cosine_score = util.pytorch_cos_sim(embedA, embedB).item()
              results[lang].append({"titleA":row['titleA'],"titleB":row['titleB'], "predicted" : (cosine_score+1) / 2.0, "actual" : row['relatedness']/10.0})
        self.save_similarity_scores(results)
        return results

    def save_metrics_results(self, Average_of_true_ranks, r_s):
        with open("WikiSRS/results/"+self.dataset_name+"/sbert_metrics_results", "w") as f :
            f.write("Average of True Rankings for each language : \n")
            for k, v in Average_of_true_ranks.items():
                f.write('%s: %s\n' % (k, v))
            f.write("\nSpearman Correlation Results for each language : \n")
            for k, v in r_s.items():
                f.write('%s: %s\n' % (k, v))
        return

    def evaluate_ranking(self):
        true_rank = {'en':0,'ar':0,'es':0,'fr':0,'ru':0,'pt':0}
        Average_of_true_ranks = {'en':0,'ar':0,'es':0,'fr':0,'ru':0,'pt':0}
        counter = {'en':0,'ar':0,'es':0,'fr':0,'ru':0,'pt':0}
        r_s = {'en':0,'ar':0,'es':0,'fr':0,'ru':0,'pt':0}

        for lang in self.wiki_languages:
            lang_results = self.similarity_scores[lang]
            for rowA in lang_results:
                for rowB in lang_results:
                    if rowA['titleA'] == rowB['titleA'] and rowA['titleB'] == rowB['titleB']: 
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
