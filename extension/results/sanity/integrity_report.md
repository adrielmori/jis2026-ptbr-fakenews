# Integrity report

## Embedding matrices

| encoder             | split   |   n_rows |   dim | dtype   |   n_meta | finite_head   |
|:--------------------|:--------|---------:|------:|:--------|---------:|:--------------|
| bertimbau_base      | train   |    29311 |   768 | float32 |    29311 | True          |
| bertimbau_base      | val     |     6281 |   768 | float32 |     6281 | True          |
| bertimbau_base      | test    |     6281 |   768 | float32 |     6281 | True          |
| bertimbau_large     | train   |    29311 |  1024 | float32 |    29311 | True          |
| bertimbau_large     | val     |     6281 |  1024 | float32 |     6281 | True          |
| bertimbau_large     | test    |     6281 |  1024 | float32 |     6281 | True          |
| albertina_ptbr_100m | train   |    29311 |   768 | float32 |    29311 | True          |
| albertina_ptbr_100m | val     |     6281 |   768 | float32 |     6281 | True          |
| albertina_ptbr_100m | test    |     6281 |   768 | float32 |     6281 | True          |
| albertina_ptbr_900m | train   |    29311 |  1536 | float32 |    29311 | True          |
| albertina_ptbr_900m | val     |     6281 |  1536 | float32 |     6281 | True          |
| albertina_ptbr_900m | test    |     6281 |  1536 | float32 |     6281 | True          |

## Per-corpus counts (embeddings metadata vs unified dataset)

| dataset_id   |   n_embedded |   n_unified |   n_train |   n_val |   n_test |   n_fake |   n_real | embedded   | match_unified   |
|:-------------|-------------:|------------:|----------:|--------:|---------:|---------:|---------:|:-----------|:----------------|
| BOATOSBR     |            0 |        3404 |         0 |       0 |        0 |        0 |        0 | False      | False           |
| CENTRALFATOS |        10316 |       10316 |      7222 |    1547 |     1547 |    10282 |       34 | True       | True            |
| COVID19BR    |         2399 |        2399 |      1679 |     360 |      360 |      905 |     1494 | True       | True            |
| FACTCKBR     |         2106 |        2106 |      1474 |     316 |      316 |     1866 |      240 | True       | True            |
| FAKEBR       |         7200 |        7200 |      5040 |    1080 |     1080 |     3600 |     3600 | True       | True            |
| FAKETRUEBR   |         3582 |        3582 |      2508 |     537 |      537 |     1791 |     1791 | True       | True            |
| FCN          |         2064 |        2064 |      1445 |     310 |      309 |     1044 |     1020 | True       | True            |
| FNEWSSET     |          600 |         600 |       420 |      90 |       90 |      300 |      300 | True       | True            |
| FRECOGNA     |        11902 |       11902 |      8331 |    1785 |     1786 |     5951 |     5951 | True       | True            |
| MUMINPT      |         1404 |        1404 |       982 |     211 |      211 |     1339 |       65 | True       | True            |
| TRE300       |          300 |         300 |       210 |      45 |       45 |      148 |      152 | True       | True            |

Split files (codes/results/splits) match the embedding metadata: yes

## Selected hyper-parameters

Pairs found: 32 / 32. Missing: none

| encoder             | classifier        | best_params                                                                     |
|:--------------------|:------------------|:--------------------------------------------------------------------------------|
| albertina_ptbr_900m | svc_rbf           | {'clf__C': 10.0, 'clf__gamma': 'scale'}                                         |
| bertimbau_large     | svc_rbf           | {'clf__C': 10.0, 'clf__gamma': 'scale'}                                         |
| bertimbau_base      | svc_rbf           | {'clf__C': 10.0, 'clf__gamma': 'scale'}                                         |
| albertina_ptbr_100m | svc_rbf           | {'clf__C': 10.0, 'clf__gamma': 'scale'}                                         |
| albertina_ptbr_900m | linearsvc         | {'clf__C': 0.01}                                                                |
| albertina_ptbr_900m | logreg            | {'clf__C': 0.01, 'clf__solver': 'liblinear'}                                    |
| albertina_ptbr_900m | knn               | {'clf__metric': 'minkowski', 'clf__n_neighbors': 7, 'clf__weights': 'distance'} |
| bertimbau_large     | linearsvc         | {'clf__C': 0.01}                                                                |
| bertimbau_large     | logreg            | {'clf__C': 0.01, 'clf__solver': 'liblinear'}                                    |
| bertimbau_base      | knn               | {'clf__metric': 'minkowski', 'clf__n_neighbors': 5, 'clf__weights': 'distance'} |
| albertina_ptbr_900m | gradient_boosting | {'clf__learning_rate': 0.1, 'clf__max_depth': 3, 'clf__n_estimators': 200}      |
| bertimbau_large     | knn               | {'clf__metric': 'minkowski', 'clf__n_neighbors': 5, 'clf__weights': 'distance'} |
| bertimbau_large     | gradient_boosting | {'clf__learning_rate': 0.1, 'clf__max_depth': 3, 'clf__n_estimators': 200}      |
| albertina_ptbr_100m | knn               | {'clf__metric': 'minkowski', 'clf__n_neighbors': 7, 'clf__weights': 'distance'} |
| albertina_ptbr_900m | random_forest     | {'clf__max_depth': None, 'clf__min_samples_split': 2, 'clf__n_estimators': 500} |
| bertimbau_large     | random_forest     | {'clf__max_depth': None, 'clf__min_samples_split': 2, 'clf__n_estimators': 500} |
| bertimbau_base      | linearsvc         | {'clf__C': 0.01}                                                                |
| bertimbau_base      | logreg            | {'clf__C': 0.1, 'clf__solver': 'liblinear'}                                     |
| bertimbau_base      | gradient_boosting | {'clf__learning_rate': 0.1, 'clf__max_depth': 3, 'clf__n_estimators': 200}      |
| bertimbau_base      | random_forest     | {'clf__max_depth': None, 'clf__min_samples_split': 2, 'clf__n_estimators': 500} |
| albertina_ptbr_100m | logreg            | {'clf__C': 0.1, 'clf__solver': 'liblinear'}                                     |
| albertina_ptbr_100m | linearsvc         | {'clf__C': 0.1}                                                                 |
| albertina_ptbr_100m | gradient_boosting | {'clf__learning_rate': 0.1, 'clf__max_depth': 3, 'clf__n_estimators': 200}      |
| albertina_ptbr_100m | random_forest     | {'clf__max_depth': None, 'clf__min_samples_split': 2, 'clf__n_estimators': 500} |
| albertina_ptbr_900m | decision_tree     | {'clf__criterion': 'gini', 'clf__max_depth': 10, 'clf__min_samples_split': 2}   |
| bertimbau_base      | decision_tree     | {'clf__criterion': 'gini', 'clf__max_depth': 10, 'clf__min_samples_split': 2}   |
| bertimbau_large     | decision_tree     | {'clf__criterion': 'gini', 'clf__max_depth': 10, 'clf__min_samples_split': 5}   |
| albertina_ptbr_100m | decision_tree     | {'clf__criterion': 'gini', 'clf__max_depth': 10, 'clf__min_samples_split': 2}   |
| bertimbau_base      | gaussian_nb       | {'clf__var_smoothing': 1e-09}                                                   |
| bertimbau_large     | gaussian_nb       | {'clf__var_smoothing': 1e-09}                                                   |
| albertina_ptbr_100m | gaussian_nb       | {'clf__var_smoothing': 1e-09}                                                   |
| albertina_ptbr_900m | gaussian_nb       | {'clf__var_smoothing': 1e-09}                                                   |

## Published per-item test predictions

models: 32; rows per model: min=6281 max=6281

## Divergences / notes

- BOATOSBR: present in the unified dataset (3404 rows) but never embedded (not part of the published 10-corpus benchmark)