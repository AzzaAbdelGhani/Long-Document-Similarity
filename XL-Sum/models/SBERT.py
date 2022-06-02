import torch
import faiss
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


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class SBERT_XL_Sum() : 
    def __init__(self, dataset_name, save=False, saved_embeddings_path = None):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.wiki_languages = ['english','arabic','spanish','french','russian','portuguese']
        self.dataset_name = dataset_name
        self.data = self.load_dataset(dataset_name)
        if saved_embeddings_path == None:
            self.model = SentenceTransformer('distiluse-base-multilingual-cased')
            self.model.max_seq_length = self.model.get_max_seq_length()
            self.data_embeddings = self.compute_text_embeddings()
            if save==True:
                with open('data/sbert_embeddings/'+dataset_name+'_embeddings.pkl', "wb") as fOut:
                    pickle.dump(self.data_embeddings, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            if self.device == "cpu":
                with open(saved_embeddings_path, "rb") as f:
                    self.data_embeddings = CPU_Unpickler(f).load()
            else:
                self.data_embeddings = pd.read_pickle(saved_embeddings_path) 

        self.results = self.test_model()
        self.MRR = self.evaluate_results()

    def load_dataset(self,dataset_name):
        dataset = pd.read_csv('data/'+dataset_name+'_dataset.csv')
        return dataset
    
    def split_sentence(self,sentence):
        if len(sentence.split()) > 120:
            s = []
            s.append(' '.join(sentence.split()[:120]))
            s.append(' '.join(self.split_sentence(' '.join(sentence.split()[120:]))))
            return s
        else: 
            return [sentence]

    def compute_text_embeddings(self):
        embeddings = {}
        for language in self.wiki_languages:
            embeddings[language] = []
        for id, row in tqdm(self.data.iterrows(), desc="compute Embeddings: "):
            summary_sentences = [row.title]
            text_sentences = [row.title]
            for sentence in sent_tokenize(row.summary):
                s = self.split_sentence(sentence)
                summary_sentences.extend(s)
            summary_embeddings = self.model.encode(summary_sentences, convert_to_tensor=True, show_progress_bar=False)
            for sentence in sent_tokenize(row.text):
                s = self.split_sentence(sentence)
                text_sentences .extend(s)
            text_embeddings = self.model.encode(text_sentences, convert_to_tensor=True, show_progress_bar=False)
            embeddings[row.lang].append({"url": row.url, "summary_embedding": torch.mean(summary_embeddings, dim=0), "text_embedding": torch.mean(text_embeddings, dim=0)}) 
        return embeddings

    def faiss_index(self,query_idx,query_lang, query, k):
        res = faiss.StandardGpuResources()
        d = len(self.data_embeddings[query_lang][query_idx]['text_embedding']) #embedding's size 
        n = len(self.data_embeddings[query_lang]) #number of articles
        articles_with_same_language = [item['url'] for item in self.data_embeddings[query_lang]]
        other_articles_embeddings = [item['summary_embedding'] for item in self.data_embeddings[query_lang]]
        document_embeddings = numpy.array([numpy.array(x.cpu()) for x in other_articles_embeddings])
        
        index = faiss.IndexFlatL2(d)   # build the index, d=size of vectors 
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(document_embeddings)                  
        D, I = gpu_index.search(query, k) 
        I_titles = [] 
        for i in I[0]:
            I_titles.append(articles_with_same_language[i])
        return(I_titles)

    def find_similar_summaries(self, lang, url, num_items):
        #print([d['url'] for d in self.data_embeddings[lang]])
        idx = [d['url'] for d in self.data_embeddings[lang]].index(url)
        query_embed = self.data_embeddings[lang][idx]['text_embedding'].cpu().numpy().reshape(1,len(self.data_embeddings[lang][idx]['text_embedding']))
        similar_docs = self.faiss_index(idx, query_lang=lang, query=query_embed, k=num_items)
        return similar_docs

    def test_model(self, k=1000):
        model_labels = {}
        for language in self.wiki_languages:
            model_labels[language] = {}
        for lang, url in tqdm(zip(list(self.data.lang), list(self.data.url)), desc="Find k = {} Similar articles".format(k)):
            #print(lang)
            #print(url)
            model_labels[lang][url] = self.find_similar_summaries(lang, url, k)
        return model_labels

    def save_MRR_results(self, MRR):
        with open("results/"+self.dataset_name+"/sbert_MRR.txt", "w") as f:
            for key, value in MRR.items():
                f.write("{} : {:.1%}\n".format(key, value))
                #f.write('%s : %s\n' % (key, value))
        return
    
    def evaluate_results(self):
        reciprocal_ranks = {}
        MRR = {}
        for lang in tqdm(self.wiki_languages, desc="Compute MRR for each language"):
            reciprocal_ranks[lang] = []
            for doc, res in self.results[lang].items():
                doc_rank = []
                if doc in res :
                    doc_rank.append(res.index(doc))
                if len(doc_rank) == 0 : 
                    reciprocal_ranks[lang].append(0)
                elif min(doc_rank) == 0 :
                    reciprocal_ranks[lang].append(1)
                else:
                    reciprocal_ranks[lang].append(1/min(doc_rank))

            MRR[lang] = sum(reciprocal_ranks[lang])/len(reciprocal_ranks[lang])
            #print(f"Recepricle mean: {sum(reciprocal_ranks) / len(reciprocal_ranks)}\n")
        self.save_MRR_results(MRR)
        return MRR
