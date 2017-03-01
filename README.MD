This project was made as part of an Applied Machine Learning class with Serge Belongie

HW1:

1. Trained a k-nearest-neighbors algorithm to recognize sketches of digits and return identity of new digits in test set.
2. Trained a- algorithm on a list of Titanic passengers. Then, using demographics and type of cabin from the training dataset, the algorithm evaluated whether a hiterto-unseen list of passengers survived the Titanic disaster or not.

HW2:

1. Using the Yale Eigenfaces database, we calculated an “average” face, performed a Singular Value Decomposition, found the low-rank approximation, plotted the rank-r approximation error, represented the 2500-dimensional face image using an r-dimensional feature vector and used logistic regression to classify features from the top r vectors.

2. Using Kaggle’s “What’s Cooking” recipe database, we represented each dish by a binary ingredient feature vector, used Naive Bayes Classifier to predict cuisine using recipe ingredients, calculated the Naive Bayes’ accuracy assuming Gaussian and Bernoulli priors, used Logistic Regression to predict cuisine using recipe ingredients, and finally used our Bernoulli-prior Naive Bayes to compete in the Kaggle contest, where we reached 85% accuracy.


HW3:

1. This assignment performs sentiment analysis of online reviews on Amazon, Yelp and IMDB. We preprocessed the written data (punctuation stripping, lemmatization, etc), represented each of the 3 review collections in a Bag of Words model, normalized the Bag of Words, implemented K-means to divide the training set into a “positive” and a “negative” cluster, and then used a logistic regression model to predict the review’s sentiment. We reached an accuracy of 81.5%. We then ran the same process using a 2-gram n-model instead of Bag of Words; using a logistic regression model, we reached 66.7% accuracy in sentiment prediction. Finally, we implemented PCA to reduce the dimensions of features and then implemented Bag of Words on the reduced model. 

2. We used the Old Faitful Geyser dataset, which contains 272 observation of the geyser’s eruption time and waiting time. We implemented a bimodal Gaussian Mixture model and calculated how many iterations were necessary for a convergence of datapoint around different covariance means. Finally, we compared a K-means algorithm to our EMM and found that it required less iterations than the GMM to cluster the data.

HW4:

1.
