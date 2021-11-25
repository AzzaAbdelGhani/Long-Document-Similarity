from data import Load_dataset


def main():
    dataset = Load_dataset.WikipediaLongDocumentSimilarityDataset("video_games") 
    #print(len(dataset.articles))
    #print(len(dataset.articles_embeddings))
    print(dataset.articles_embeddings[0])
    
    
    
if __name__ == "__main__":
    main()
