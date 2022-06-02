#The reference code for this class can be found in this post : 
# https://medium.com/red-buffer/doc2vec-computing-similarity-between-the-documents-47daf6c828cd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk
nltk.download('punkt')
from tqdm import tqdm
import re
import faiss
import numpy
from nltk.stem import SnowballStemmer
from tashaphyne.stemming import ArabicLightStemmer
import spacy
from scipy import stats
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from spacy.lang.ar.stop_words import STOP_WORDS as ar_stop
from spacy.lang.es.stop_words import STOP_WORDS as es_stop
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.ru.stop_words import STOP_WORDS as ru_stop
from spacy.lang.pt.stop_words import STOP_WORDS as pt_stop


class Doc2Vec_XLSumSimilarity():
    def __init__(self, dataset_name=None, clean=False, train=False, test=False, save=False):
        self.dataset_name = dataset_name
        self.wiki_languages = ['english','arabic','spanish','french','russian','portuguese']
        self.data = self.load_dataset(dataset_name, clean)
        self.doc2vec_models = {}
        if train == True : 
            self.doc2vec_models = self.train_doc2vec_model()
            if save == True:
                for l in self.wiki_languages:
                    self.doc2vec_models[l].save("data/doc2vec_models/"+l+"_d2v.model")
                
            self.similarity_scores, self.similar_docs = self.get_most_similar_docs(num_docs=1000)

        if test == True: 
            for lang in self.wiki_languages:
                self.doc2vec_models[lang] = Doc2Vec.load("data/doc2vec_models/"+lang+"_d2v.model")
            self.inferred_vectors = self.get_inferred_vectors()
            self.similar_docs = self.get_similar_summaries(k=1000)


        self.MRR = self.compute_MRR(self.similar_docs)

    def load_dataset(self,dataset_name, clean):
        if clean == True : 
            dataset = pd.read_csv('data/'+dataset_name+'_dataset.csv')
            dataset.rename(columns={'Unnamed: 0': 'idx'}, inplace=True)
            #dataset = dataset[:150]
            dataset['clean_text'], dataset['clean_summary'] = self.cleaning_data(dataset)
            dataset.to_csv("data/"+"clean_"+dataset_name+"_data.csv")
        else :
            dataset = pd.read_csv("data/"+"clean_"+dataset_name+"_data.csv")
            #dataset = pd.read_csv("/kaggle/working/clean_"+dataset_name+"_data.csv")
        dataset.fillna('', inplace=True)
        return dataset
    
    def cleaning_data(self, data):
        clean_text = []
        clean_summary = []
        for lang, text, summary in tqdm(zip(list(data.lang), list(data.text), list(data.summary)), desc="Cleaning dataset "):
            clean_text.append(self.preprocessing(lang,text))
            clean_summary.append(self.preprocessing(lang,summary))
        return clean_text, clean_summary
        
        
    def preprocessing(self,language,content):
        multilingual_stopwords = list(en_stop)+list(ar_stop)+list(es_stop)+list(fr_stop)+list(ru_stop)+list(pt_stop)
        text = ' '.join([c.lower() for c in content.split() if c not in multilingual_stopwords])
        punctiation_pattern = re.compile('[!-_@#$%^&*()?<>;\.,:"]')
        text = re.sub(punctiation_pattern, '', text)
        numbers_patterns = re.compile('[0-9]+[\w]*')
        text = re.sub(numbers_patterns, '', text)
        clean_text = text
        if language == 'arabic':
            ArListem = ArabicLightStemmer()
            clean_text = ' '.join([ArListem.light_stem(token) for token in text.split(" ")])
        else :
            stemmer = SnowballStemmer(language)
            clean_text = ' '.join([stemmer.stem(token) for token in text.split(" ")])

        return clean_text
    
    def tokenize_data(self): 
        tokenized_text = {}
        for lang in self.wiki_languages:
            tokenized_text[lang] = {}

        for url,lang,clean_text,clean_summary in tqdm(zip(list(self.data.url), list(self.data.lang), list(self.data.clean_text), list(self.data.clean_summary)),desc="Tokenizing Data "):
            tokenized_text[lang]["text_"+url] = word_tokenize(clean_text)
            tokenized_text[lang]["summary_"+url] = word_tokenize(clean_summary)

        return tokenized_text
    

    def train_doc2vec_model(self):
        '''
        Tagging the data
        Initializing doc2vec
        Building the vocabulary of tagged data
        Training doc2vec
        '''
        tokenized_data = self.tokenize_data()
        tagged_pages = {}
        models = {}
        
        for lang in tqdm(self.wiki_languages, desc="Training Doc2vec "):
            tagged_pages[lang] = [TaggedDocument(words=_d, tags=[t]) for t,_d in tokenized_data[lang].items()]
            m = Doc2Vec(vector_size=768, min_count=2, epochs=10)
            m.build_vocab(tagged_pages[lang])
            m.train(tagged_pages[lang], total_examples=m.corpus_count, epochs=10)
            models[lang] = m
            print("{} Doc2vec model training is done!\n".format(lang))
        return models
    

    def save_similar_docs_results(self, sim_docs):
        with open("results/"+self.dataset_name+"/doc2vec_similar_summaries.txt", "w") as f:
            for key, value in sim_docs.items():
                f.write('%s :\n' % key)
                for ik, iv in value.items():
                    f.write('%s : %s\n' % (ik, iv))
        return

    def get_most_similar_docs(self, num_docs = 1000):
        similarity_scores = []
        similar_docs = {}
        for lang in self.wiki_languages:
            similar_docs[lang] = {}

        for i, row in tqdm(self.data.iterrows(), desc="Find Similar Summaries for each article "):
            similarity_scores.append(self.doc2vec_models[row.lang].docvecs.distance("text_"+row.url, "summary_"+row.url))
            similar_docs[row.lang][row.url] = self.doc2vec_models[row.lang].docvecs.most_similar('text_'+row.url, topn=num_docs)
        #self.save_similar_docs_results(similar_docs)       
        return similarity_scores, similar_docs


    def get_inferred_vectors(self):
        inferred_vectors = {}
        for language in self.wiki_languages:
            inferred_vectors[language] = []
        for url, lang, clean_text, clean_summary in tqdm(zip(list(self.data.url), list(self.data.lang), list(self.data.clean_text), list(self.data.clean_summary)), desc="Get Inferred Vectors "):
            text_vector = self.doc2vec_models[lang].infer_vector(word_tokenize(clean_text))
            summary_vector = self.doc2vec_models[lang].infer_vector(word_tokenize(clean_summary))
            inferred_vectors[lang].append({"url": url, "summary_vector": summary_vector, "text_vector": text_vector})
        return inferred_vectors


    def faiss_index(self, data_embeddings, query_idx, query, k):
        res = faiss.StandardGpuResources()
        #keys = list(data_embeddings.keys())
        #values = list(data_embeddings.values())
        d = len(data_embeddings[query_idx]['text_vector']) #embedding's size 
        n = len(data_embeddings) #number of articles

        summaries_embeddings = numpy.array([numpy.array(x) for x in [v['summary_vector'] for v in data_embeddings]])
        
        index = faiss.IndexFlatL2(d)   # build the index, d=size of vectors 
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(summaries_embeddings)                  
        D, I = gpu_index.search(query, k) 
        I_titles = [] 
        for i in I[0]:
            I_titles.append(data_embeddings[i]['url'])
        return(I_titles)


    def find_similar_summaries(self, url, lang, num_items):
        data_embeddings = self.inferred_vectors[lang]
        query_idx = [d['url'] for d in data_embeddings].index(url)
        query_embed = data_embeddings[query_idx]['text_vector'].reshape(1, len(data_embeddings[query_idx]['text_vector']))
        similar_summs = self.faiss_index(data_embeddings, query_idx, query_embed, num_items)
        return similar_summs

    def get_similar_summaries(self, k=1000):
        model_similarities = {}
        for lang in self.wiki_languages:
            model_similarities[lang] = {}
        for url, lang in tqdm(zip(list(self.data.url), list(self.data.lang)), desc="Find top k = {} Similar summaries".format(k)):
            model_similarities[lang][url] = self.find_similar_summaries(url,lang, k)
        return model_similarities

    def save_MRR_results(self, MRR):
        with open("results/"+self.dataset_name+"/doc2vec_MRR.txt", "w") as f:
            for key, value in MRR.items():
                f.write("{} : {:.1%}\n".format(key, value))
                #f.write('%s : %s\n' % (key, value))
        return

    def compute_MRR(self, sim_docs):
        reciprocal_ranks = {}
        MRR = {}
        for lang in tqdm(self.wiki_languages, desc="Compute MRR for each language"):
            reciprocal_ranks[lang] = []
            for doc, res in sim_docs[lang].items():
                doc_rank = []
                for item in res : 
                    if item == doc:
                        doc_rank.append(res.index(item))
                if len(doc_rank) == 0 : 
                    reciprocal_ranks[lang].append(0)
                elif min(doc_rank) == 0 :
                    reciprocal_ranks[lang].append(1)
                else:
                    reciprocal_ranks[lang].append(1/min(doc_rank))

            MRR[lang] = sum(reciprocal_ranks[lang])/len(reciprocal_ranks[lang])
        self.save_MRR_results(MRR)
        #print("{:.1%}".format(MRR))
        return MRR

