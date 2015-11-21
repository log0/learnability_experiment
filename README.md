# Experiment on Learnability of a Model on Games

This is an exploratory project to see how well a machine learning, or more precisely a particular machine learning model, learns how to play a simple game by learning the rules hidden within. At this stage, I am only looking to have the model learn extremely simple games (in human sense). I will explore if a model can do better to learn more complex games.

One of the goal here is that I want to avoid, by will also do so to learn more insight:
1) doing heavy feature engineering;
2) doing reinforcement learning;

This is just meant for my notes so the notes will be quite non-linear for reading. Refer to the commit dates to help you navigate.

So far the experiments are conducted by Python Scikit-Learn's Random Forest Classifier model, since this is one of the models in the library that supports predicting a vector of output. However, without looking at the code, I suspect the RandomForestClassifier doesn't treat the vector of output as a series of related variables.