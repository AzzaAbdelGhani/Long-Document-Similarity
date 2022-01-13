from data import Load_dataset
from models.SBERT import SBERT
from models.evaluation_metrics import Evaluate_model
from models.TFIDF import TF_IDF
from tabulate import tabulate 

def main():
    table = []
    sbert_model_1 = SBERT("video_games", saved_embeddings= "data/saved_embeddings/all-MiniLM-L6-v2_video_games_embeddings.pkl")
    m1 = []
    m1.append("SBERT_all_MiniLM")
    res1 = Evaluate_model(sbert_model_1, k=100)
    m1.extend(res1)
    sbert_model_2 = SBERT("wines", saved_embeddings= "data/saved_embeddings/all-MiniLM-L6-v2_wines_embeddings.pkl")
    res2 = Evaluate_model(sbert_model_2, k=100)
    m1.extend(res2)
    table.append(m1)

    sbert_model_3 = SBERT("video_games", saved_embeddings= "data/saved_embeddings/all-mpnet-base-v2-video_games_embeddings.pkl")
    m1 = []
    m1.append("SBERT_all_mpnet")
    res7 = Evaluate_model(sbert_model_3, k=100)
    m1.extend(res7)
    sbert_model_4 = SBERT("wines", saved_embeddings= "data/saved_embeddings/all-mpnet-base-v2-wines_embeddings.pkl")
    res8 = Evaluate_model(sbert_model_4, k=100)
    m1.extend(res8)
    table.append(m1)

    sbert_model_5 = SBERT("video_games", saved_embeddings= "data/saved_embeddings/all-roberta-large-v1-video_games_embeddings.pkl")
    m1 = []
    m1.append("SBERT_all_roberta")
    res9 = Evaluate_model(sbert_model_5, k=100)
    m1.extend(res9)
    sbert_model_6 = SBERT("wines", saved_embeddings= "data/saved_embeddings/all-roberta-large-v1-wines_embeddings.pkl")
    res10 = Evaluate_model(sbert_model_6, k=100)
    m1.extend(res10)
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
    m3.append("Ext_Summ(100)+TFIDF")
    res5 = Evaluate_model(sum_tfidf_model_1 , k=100)
    m3.extend(res5)

    sum_tfidf_model_2 = TF_IDF("wines", use_summarization=True)
    res6 = Evaluate_model(sum_tfidf_model_2 , k=100)
    m3.extend(res6)
    table.append(m3)


    headers=["Model","MPR", "MRR", "HR@100", "MPR", "MRR", "HR@100"]
    print("\n Results : \n")
    print("-----------------------------------------------------------------------------")
    print("| \t\t      |   \t video_games  \t |   \t wines \t            |")
    print("-----------------------------------------------------------------------------")
    print(tabulate(table, headers, tablefmt="github"))

    
if __name__ == "__main__":
    main()
