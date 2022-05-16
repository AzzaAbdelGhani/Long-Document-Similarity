# Long-Document-Similarity

### Results

In this folder, a set of baseline models were implemented to estimate the semantic similarity between long-text Wikipedia articles by using two Wikipedia datsets, namely Wikipedia movies dataset and Wikipedia video-games datset. Each model has different strategy to represent articles, first model is TF-IDF, where each article is represented as a vector of the words that it contains with a tf-idf weightning score. Another way is to apply Extractive Summarization first and then use TF-IDF. Finally by SBERT model to get sentence embedding for each sentence in the article and then average these embeddings to get the total article embedding. 


To evaluate the results, three evaluatoin metrics are computed MPR(Mean Percentile Rank), MRR(Mean Reciprocal Rank) and HR@k(Hit Ratio at k). At the end, we compare our results with SDR model (stated in [this paper](https://arxiv.org/abs/2106.01186)), as shown in this table : 


<p align="center">
    <img src="data/images/results.png" width="700"/>
</p>

