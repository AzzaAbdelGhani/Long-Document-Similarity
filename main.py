from data import Load_dataset
from models import SBERT


def main():
    sdr_model = SBERT.SBERT("video_games")
    dataset_articles = sdr_model.dataset
    dataset_embeddings = sdr_model.articles_embeddings
    print(len(dataset_articles))
    print(len(dataset_embeddings))
    
    
    
if __name__ == "__main__":
    main()
