from data import Load_dataset
from models.SBERT import SBERT


def main():
    sbert_model = SBERT("video_games", saved_embeddings= "data/video_games_embeddings.pkl")
    result = sbert_model.find_similar_docs("Dead Island", 14)
    print(result)
    
    
    
if __name__ == "__main__":
    main()
