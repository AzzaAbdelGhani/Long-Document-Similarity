from data import Load_dataset
from models import SBERT


def main():
    sbert_model = SBERT.SBERT("video_games")
    result = sbert_model.find_similar_docs("Dead Island", 13)
    
    
    
if __name__ == "__main__":
    main()
