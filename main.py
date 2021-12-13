from data import Load_dataset
from models.SBERT import SBERT
from models.evaluation_metrics import Evaluate_model
from models.TFIDF import TF_IDF

def main():
    sbert_model_1 = SBERT("video_games", saved_embeddings= "data/saved_embeddings/video_games_embeddings.pkl")
    MPR, MRR, HIT_RATIO_AT_100 = Evaluate_model(sbert_model_1, k=100)
    print("\nSBERT - video_games dataset's results :")
    print("Mean Percentile Rank : {0:.1%}\n".format(MPR))
    print("Mean Reciprocal Rank : {0:.1%}\n".format(MRR))
    print("Mean Hit Rate@100 : {0:.1%}\n".format(HIT_RATIO_AT_100))
    
    sbert_model_2 = SBERT("wines", saved_embeddings= "data/saved_embeddings/wines_embeddings.pkl")
    MPR, MRR, HIT_RATIO_AT_100 = Evaluate_model(sbert_model_2, k=100)
    print("\nSBERT - wines dataset's results :")
    print("Mean Percentile Rank : {0:.1%}\n".format(MPR))
    print("Mean Reciprocal Rank : {0:.1%}\n".format(MRR))
    print("Mean Hit Rate@100 : {0:.1%}\n".format(HIT_RATIO_AT_100))

    tfidf_model_1 = TF_IDF("video_games")
    MPR, MRR, HIT_RATIO_AT_100 = Evaluate_model(tfidf_model_1, k=100)
    print("\nTFIDF - video_games dataset's results :")
    print("Mean Percentile Rank : {0:.1%}\n".format(MPR))
    print("Mean Reciprocal Rank : {0:.1%}\n".format(MRR))
    print("Mean Hit Rate@100 : {0:.1%}\n".format(HIT_RATIO_AT_100))

    tfidf_model_2 = TF_IDF("wines")
    MPR, MRR, HIT_RATIO_AT_100 = Evaluate_model(tfidf_model_2, k=100)
    print("\nTFIDF - wines dataset's results :")
    print("Mean Percentile Rank : {0:.1%}\n".format(MPR))
    print("Mean Reciprocal Rank : {0:.1%}\n".format(MRR))
    print("Mean Hit Rate@100 : {0:.1%}\n".format(HIT_RATIO_AT_100))
    

    
if __name__ == "__main__":
    main()
