def Evaluate_model(model, k):
  MPR = calculate_MPR(model.dataset.labels, model.results, len(model.dataset))
  MRR = calculate_MRR(model.dataset.labels, model.results)
  Hit_Ratio = calculate_Hit_Ratio_at_K(model.dataset.labels, model.test_model(k))
  return MPR, MRR, Hit_Ratio

def calculate_MPR(gt_labels, model_labels, num_articles):
  percentiles = []
  for doc, labels in gt_labels.items():
    for label, count in labels.items():
      if label not in model_labels[doc]:
        continue
      label_rank = model_labels[doc].index(label)
      percentiles.append((label_rank/num_articles)*count)
  MPR = sum(percentiles)/len(percentiles)
  #print("percentiles_mean:{}\n\n\n\n".format(sum(percentiles) / len(percentiles)))
  return MPR

def calculate_MRR(gt_labels, model_labels):
  reciprocal_ranks = []
  for doc, labels in gt_labels.items():
    ranks = []
    for label in labels.keys():
      if label not in model_labels[doc]:
        continue
      label_rank = model_labels[doc].index(label)
      ranks.append(label_rank)
    if len(ranks) == 0 :
      continue 
    reciprocal_ranks.append(1/min(ranks))
  MRR = sum(reciprocal_ranks)/len(reciprocal_ranks)
  #print(f"Recepricle mean:{sum(reciprocal_ranks) / len(reciprocal_ranks)}")
  return MRR

def calculate_Hit_Ratio_at_K(gt_labels, model_labels_at_k):
  mean_ratios = []
  for doc, labels in gt_labels.items():
    similar_articles_found = 0
    for label in labels.keys():
      if label in model_labels_at_k[doc]:
        similar_articles_found = similar_articles_found + 1 
    mean_ratios.append((similar_articles_found/len(labels.keys())))
  mean_hit_ratio = sum(mean_ratios)/len(mean_ratios)
  #print(f"Hit rate mean:{mean_hit_ratio}")
  return mean_hit_ratio
