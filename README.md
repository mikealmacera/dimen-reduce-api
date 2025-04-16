# dimen-reduce-api

REST API for computing (Truncated) SVD and PCA. 


## How To Use

POST your matrix in JSON. (2D only) @ /tsvd or /pca

You will be returned your matrix ID. 

![tsvdpost](https://github.com/user-attachments/assets/5e728e04-6dbc-4455-822a-17f2f35b1529)

GET your SVD or PCA by inputing your request @ /tsvd/<id> or /pca/<id>. You can include a query parameter for your rank `?rank=N` (positive integers ONLY).

You will be returned necessary U, \Sigma, V^T for TSVD

"   "  "  returned necessary components, explained variance, explained variance ratio for PCA.

![tsvdget](https://github.com/user-attachments/assets/02d730e5-fe04-41be-9f8d-cb21417ae372)
