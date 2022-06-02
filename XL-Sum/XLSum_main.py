from models.Doc2vec import Doc2Vec_XLSumSimilarity
from models.SBERT import SBERT_XL_Sum

def main():

    ############# SBERT #################
    # First we need to get sbert documents embeddings and save them in data/sbert_embeddings folder:

    train_sbert = SBERT_XL_Sum(dataset_name="train", save=True)
    test_sbert = SBERT_XL_Sum(dataset_name="test", save=True)
    val_sbert = SBERT_XL_Sum(dataset_name="val", save=True)

    # If the embeddings are already saved, then we can only load them and then get the results : 
    #train_sbert = SBERT_XL_Sum(dataset_name="train", saved_embeddings_path='data/sbert_embeddings/train_embeddings.pkl')
    #test_sbert = SBERT_XL_Sum(dataset_name="test", saved_embeddings_path='data/sbert_embeddings/test_embeddings.pkl')
    #val_sbert = SBERT_XL_Sum(dataset_name="val", saved_embeddings_path='data/sbert_embeddings/val_embeddings.pkl')


    ############# Doc2Vec #################

    # First, we need to train Doc2Vec models on each language and save the models in data/doc2vec_models folder 
    # also, data need to be cleaned and then saved in data folder 
    print("\nStart Training docevec models on each language \n")
    train_doc2vec = Doc2Vec_XLSumSimilarity(dataset_name="train", clean=True, train=True, save=True)
    print("\nTest doc2vec models on val dataset \n")
    val_doc2vec = Doc2Vec_XLSumSimilarity(dataset_name="val", clean=True, train=False, save=False, test=True)
    print("\nTest doc2vec models on test dataset \n")
    testing_doc2vec = Doc2Vec_XLSumSimilarity(dataset_name="test", clean=True, train=False, save=False, test=True)
    

if __name__ == "__main__":
    main()  
