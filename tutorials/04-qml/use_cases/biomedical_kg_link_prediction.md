# Use Case: Biomedical KG Link Prediction

## Pipeline

```text
Biomedical KG
  → node embeddings
  → candidate link pairs
  → pairwise features
  → quantum kernel / QSVC
  → predict likely links
```

## Metrics

- ROC-AUC
- F1
- Precision@K
- Recall@K
