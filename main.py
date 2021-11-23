from data import Load_dataset


def main():
    dataset = Load_dataset.WikipediaLongDocumentSimilarityDataset("video_games") 
    print(dataset.articles[0])
    
    
    
if __name__ == "__main__":
    main()
