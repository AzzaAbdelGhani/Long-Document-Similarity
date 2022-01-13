from data import Load_dataset
from models.SBERT import SBERT
from models.evaluation_metrics import Evaluate_model
from models.TFIDF import TF_IDF
from tabulate import tabulate 

def main():
    table = []
    sbert_model_1 = SBERT("video_games", saved_embeddings= "data/saved_embeddings/all-MiniLM-L6-v2_video_games_embeddings.pkl")
    m1 = []
    m1.append("SBERT")
    res1 = Evaluate_model(sbert_model_1, k=100)
    m1.extend(res1)
    sbert_model_2 = SBERT("wines", saved_embeddings= "data/saved_embeddings/all-MiniLM-L6-v2_wines_embeddings.pkl")
    res2 = Evaluate_model(sbert_model_2, k=100)
    m1.extend(res2)
    table.append(m1)
    tfidf_model_1 = TF_IDF("video_games")
    m2 = []
    m2.append("TF-IDF")
    res3 = Evaluate_model(tfidf_model_1, k=100)
    m2.extend(res3)

    tfidf_model_2 = TF_IDF("wines")
    res4 = Evaluate_model(tfidf_model_2, k=100)
    m2.extend(res4)
    table.append(m2)


    sum_tfidf_model_1 = TF_IDF("video_games", use_summarization=True)
    m3 = []
    m3.append("Summarization+tfidf")
    res5 = Evaluate_model(sum_tfidf_model_1 , k=100)
    m3.extend(res5)

    sum_tfidf_model_2 = TF_IDF("wines", use_summarization=True)
    res6 = Evaluate_model(sum_tfidf_model_2 , k=100)
    m3.extend(res6)
    table.append(m3)


    headers=["Model","MPR", "MRR", "HR@100", "MPR", "MRR", "HR@100"]
    print("\n Results : \n")
    print("-----------------------------------------------------------------------------")
    print("| \t\t      |   \t video_games  \t |   \t wines \t           |")
    print("-----------------------------------------------------------------------------")
    print(tabulate(table, headers, tablefmt="github"))

    
if __name__ == "__main__":
    main()
