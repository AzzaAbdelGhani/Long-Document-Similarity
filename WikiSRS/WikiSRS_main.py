from models.SBERT import SBERT_WikiSimilarity 
from models.Doc2vec import Doc2vec_WikiSimilarity

def main():

	#To get the articles embeddings and get the results : 
	#sim353  = SBERT_WikiSimilarity(dataset_name="wikipediaSimilarity353", save=True)
	#wikiSim = SBERT_WikiSimilarity(dataset_name="WikiSRS_similarity", save=True)
	#wikiRel = SBERT_WikiSimilarity(dataset_name="WikiSRS_relatedness", save=True)

	# To avid compute embeddings again and use the saved embeddings file 
	print("\nRunning SBERT on wikipediaSimilarity353 dataset : \n")
	sbert_sim353  = SBERT_WikiSimilarity(dataset_name="wikipediaSimilarity353", saved_embeddings_path="data/embeddings/wikipediaSimilarity353_embeddings.pkl")
	print("\nRunning SBERT on WikiSRS_similarity dataset : \n")
	sbert_wikiSim = SBERT_WikiSimilarity(dataset_name="WikiSRS_similarity", saved_embeddings_path="data/embeddings/WikiSRS_similarity_embeddings.pkl")
	print("\nRunning SBERT on WikiSRS_relatedness dataset : \n")
	sbert_wikiRel = SBERT_WikiSimilarity(dataset_name="WikiSRS_relatedness", saved_embeddings_path="data/embeddings/WikiSRS_relatedness_embeddings.pkl")

	print("\nTesting Doc2Vec on wikipediaSimilarity353 dataset : \n")
	doc2vec_sim353 = Doc2vec_WikiSimilarity(dataset_name="wikipediaSimilarity353", clean=True, test=True)
	print("\nTesting Doc2Vec on WikiSRS_similarity dataset : \n")
	doc2vec_SRSsim= Doc2vec_WikiSimilarity(dataset_name="WikiSRS_similarity", clean=True, test=True)
	print("\nTesting Doc2Vec on WikiSRS_relatedness dataset : \n")
	doc2vec_SRSrel = Doc2vec_WikiSimilarity(dataset_name="WikiSRS_relatedness", clean=True, test=True)

if __name__ == '__main__':
	main()