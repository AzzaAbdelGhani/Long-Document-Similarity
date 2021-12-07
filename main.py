from data import Load_dataset
from models.SBERT import SBERT
from models.evaluation_metrics import Evaluate_model


def main():
    sbert_model = SBERT("video_games", saved_embeddings= "data/saved_embeddings/video_games_embeddings.pkl")
    MPR, MRR, HIT_RATIO_AT_100 = Evaluate_model(sbert_model, k=100)
    print("Mean Percentile Rank : {}\n".format(MPR))
    print("Mean Reciprocal Rank : {}\n".format(MRR))
    print("Mean Hit Rate@100 : {}\n".format(HIT_RATIO_AT_100))
    
    
    
    
if __name__ == "__main__":
    main()
