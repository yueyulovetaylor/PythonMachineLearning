# Overview of Python Machine Learning

## Introduction

This document is a summary of how **Python Machine Learning** is structured, what each chapter is about and what learning model it introduces.

## Training Machine Learning Algorithm for Classification

After we implemented a **perceptron**, we saw how we can train **adaptive linear neurons** efficiently via a vectorized implementation of gradient descent and on-line learning via **stochastic gradient descent**. 

## A Tour of Machine Learning Classifiers Using Scikit-Learning

We have seen that **decision trees** are particularly attractive if we care about interpretability. **Logistic regression** is not only a useful model for online learning via stochastic gradient descent, but also allows us to predict the probability of a particular event. Although **support vector machines** are powerful linear models that can be extended to nonlinear problems via the kernel trick, they have many parameters that have to be tuned in order to make good predictions. In contrast, ensemble methods such as **random forests** don't require much parameter tuning and don't over t so easily as decision trees, which makes it an attractive model for many practical problem domains. The **K-nearest neighbor classifier** offers an alternative approach to classification via lazy learning that allows us to make predictions without any model training but with a more computationally expensive prediction step.

## Building Good Training Sets -- Data Preprocessing

We started this chapter by looking at useful techniques to make sure that we **handle missing data** correctly. Before we feed data to a machine learning algorithm, we also have to make sure that we **encode categorical variables** correctly, and we have seen how we can **map ordinal and nominal features values to integer** representations. We brie y discussed **L1 regularization**, which can help us to avoid overfitting by reducing the complexity of a model. As an alternative approach for removing irrelevant features, we used a **sequential feature selection algorithm** to select meaningful features from a dataset.

## Compressing Data via Dimentionality Reduction

We learned about three different, fundamental dimensionality reduction techniques for feature extraction: **standard PCA, LDA, and kernel PCA**. 

Using PCA, we projected data onto a lower-dimensional subspace to maximize the variance along the orthogonal feature axes while ignoring the class labels. 

LDA, in contrast to PCA, is a technique for **supervised dimensionality reduction**, which means that it considers class information in the training dataset to attempt to maximize the class-separability in a linear feature space. 

Lastly, we learned about a kernelized version of PCA, which allows you to **map nonlinear datasets onto a lower-dimensional feature space** where the classes become linearly separable.

## Learning Best Practices for Model Evaluation and Hyperparameter Tuning

We discussed how to chain different **transformation techniques** and classifiers in convenient model pipelines that helped us to train and evaluate machine learning models more efficiently. 

We then used those **pipelines** to perform **k-fold cross-validation**, one of the essential techniques for model selection and evaluation. Using k-fold cross-validation, we plotted learning and validation curves to diagnose the common problems of learning algorithms, such as overfitting and underfitting. 

Using **grid search**, we further fine-tuned our model. We concluded this chapter by looking at a **confusion matrix** and various different performance metrics that can be useful to further optimize a model's performance for a specific problem task.

## Combining Different Models for Ensemble Learning

Ensemble methods combine different classification models to cancel out their individual weaknesses.

We implemented a **MajorityVoteClassifier** in Python that allows us to combine different algorithms for classification. 

We then looked at **bagging**, a useful technique to reduce the variance of a model by drawing random bootstrap samples from the training set and combining the individually trained classifiers via majority vote. 

Then we discussed **AdaBoost**, which is an algorithm that is based on weak learners that subsequently learn from mistakes.

## Applying Machine Learning To Sentiment Analysis

Not only did we learn how to encode a document as a feature vector using the **bag-of-words model**, but we also learned how to weight the term frequency by relevance using **term frequency-inverse document frequency**.

## Embedding a Machine Learning Model into a Web Application

We learned how to serialize a model after training and how to load it for later use cases. Furthermore, we created a **SQLite database** for efficient data storage and created a **web application** that lets us make our movie classifier available to the outside world.

## Predict Continuous Target Variable with Regression Analysis

We built our first model by implementing linear regression using a **gradient-based optimization approach**. We then saw how to utilize scikit-learn's linear models for regression and also implement a **robust regression technique (RANSAC)** as an approach for dealing with outliers. To assess the predictive performance of regression models, we computed the mean sum of squared errors and the related R2 metric. Furthermore, we also discussed a useful graphical approach to diagnose the problems of regression models: the **residual plot**.

## Working with Unlabeled Data -- Cluster Analysis

In this chapter, you learned about three different clustering algorithms that can help us with the discovery of hidden structures or information in data. 

We started this chapter with a prototype-based approach, **k-means**, which clusters samples into spherical shapes based on a speci ed number of cluster centroids.

We then looked at a different approach to clustering: **agglomerative
hierarchical clustering**. Hierarchical clustering does not require specifying
the number of clusters upfront, and the result can be visualized in a dendrogram representation, which can help with the interpretation of the results. 

The last clustering algorithm that we saw in this chapter was **DBSCAN**, an algorithm that groups points based on local densities and is capable of handling outliers and identifying nonglobular shapes.

## Training Artificial Neural Network for Image Recognition

We learned about the most important concepts behind **multi-layer artificial neural networks**.

## Parallelizing Neural Network Training with Theano

This chapter covers a small demo of Theano and some most commonly used activation functions.
