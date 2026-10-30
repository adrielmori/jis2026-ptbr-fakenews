# Table 4 reproduction: PASS

Tolerance |diff| <= 0.005; pairs outside tolerance: 0 / 32
max |diff| = 0.00229; mean |diff| = 0.00027
mean per-item prediction agreement with published predictions: 0.99987 (min 0.99904)

| classifier        | encoder             |   published |   reproduced |         diff | within_tolerance   |
|:------------------|:--------------------|------------:|-------------:|-------------:|:-------------------|
| decision_tree     | albertina_ptbr_100m |    0.617338 |     0.617338 |  0           | True               |
| decision_tree     | albertina_ptbr_900m |    0.631501 |     0.631501 |  1.11022e-16 | True               |
| decision_tree     | bertimbau_base      |    0.606574 |     0.606574 | -1.11022e-16 | True               |
| decision_tree     | bertimbau_large     |    0.590205 |     0.590205 |  0           | True               |
| gaussian_nb       | albertina_ptbr_100m |    0.392247 |     0.392247 |  0           | True               |
| gaussian_nb       | albertina_ptbr_900m |    0.386733 |     0.386733 |  0           | True               |
| gaussian_nb       | bertimbau_base      |    0.402529 |     0.402529 |  5.55112e-17 | True               |
| gaussian_nb       | bertimbau_large     |    0.393855 |     0.393855 |  5.55112e-17 | True               |
| gradient_boosting | albertina_ptbr_100m |    0.695231 |     0.695231 |  0           | True               |
| gradient_boosting | albertina_ptbr_900m |    0.706705 |     0.706705 |  0           | True               |
| gradient_boosting | bertimbau_base      |    0.66841  |     0.66841  |  0           | True               |
| gradient_boosting | bertimbau_large     |    0.691463 |     0.691463 |  0           | True               |
| knn               | albertina_ptbr_100m |    0.715528 |     0.713388 | -0.00214002  | True               |
| knn               | albertina_ptbr_900m |    0.738883 |     0.736743 | -0.00214002  | True               |
| knn               | bertimbau_base      |    0.734604 |     0.736891 |  0.00228755  | True               |
| knn               | bertimbau_large     |    0.743864 |     0.745936 |  0.0020719   | True               |
| linearsvc         | albertina_ptbr_100m |    0.684309 |     0.684309 |  0           | True               |
| linearsvc         | albertina_ptbr_900m |    0.747864 |     0.747864 |  0           | True               |
| linearsvc         | bertimbau_base      |    0.695433 |     0.695433 |  0           | True               |
| linearsvc         | bertimbau_large     |    0.733898 |     0.733898 |  0           | True               |
| logreg            | albertina_ptbr_100m |    0.689042 |     0.688946 | -9.62214e-05 | True               |
| logreg            | albertina_ptbr_900m |    0.738085 |     0.738085 |  0           | True               |
| logreg            | bertimbau_base      |    0.694125 |     0.694125 |  0           | True               |
| logreg            | bertimbau_large     |    0.731273 |     0.731273 |  0           | True               |
| random_forest     | albertina_ptbr_100m |    0.706957 |     0.706957 |  0           | True               |
| random_forest     | albertina_ptbr_900m |    0.711405 |     0.711405 |  0           | True               |
| random_forest     | bertimbau_base      |    0.710963 |     0.710963 |  1.11022e-16 | True               |
| random_forest     | bertimbau_large     |    0.715774 |     0.715774 |  0           | True               |
| svc_rbf           | albertina_ptbr_100m |    0.791413 |     0.791413 |  0           | True               |
| svc_rbf           | albertina_ptbr_900m |    0.821896 |     0.821952 |  5.61701e-05 | True               |
| svc_rbf           | bertimbau_base      |    0.806474 |     0.806474 |  1.11022e-16 | True               |
| svc_rbf           | bertimbau_large     |    0.824253 |     0.824253 |  0           | True               |