# Tech Talent South - Introduction to Machine Learning

## Glossary of Terms

### <u>Class 1 - Machine Learning Basics</u>

- **algorithm**: the methods by which machine learning extracts insight/answers questions. More specifically, an algorithm is a sequential set of mathematical steps by which a model is trained on a specific dataset. We reviewed the Adaptive Linear Neuron (AdaLiNe) in class, which updated its weights via gradient descent with each data sample.

  ![Image result for adaline training demo](https://sebastianraschka.com/images/blog/2015/singlelayer_neural_networks_files/perceptron_animation.gif)

  GIF taken from Sebastian Raschka's blog and is based on his excellent book (that a lot of my teaching happens to come from!), *Python Machine Learning*. His blog can be found [here](https://sebastianraschka.com/blog/index.html).

- **data**: an electronic record of *something* (transaction, event, categorical, etc.), which can be structured (database tables, XML, CSV) or unstructured (log files). Data scientists analyze data using many different tools, one of which is machine learning. Data is input into an algorithm, which trains a model that makes predictions on future data.

- **data science**: an umbrella term that represents the application of scientific principles to the analysis of data. Best represented by the Venn diagram below:![img](http://2s7gjr373w3x22jf92z99mgm5w-wpengine.netdna-ssl.com/wp-content/uploads/2016/01/data-science-venn-diagram.png)

- **features**: the non-label columns of a dataset that contain predictors of the labels. Example: petal length and sepal length in the Iris dataset.

- **machine learning**: the concept of having computers analyze data, perform analyses, and make decisions without human intervention; more broadly, it's the application of algorithms to extract meaningful insights from data.

  - Anomaly detection - the application of machine learning to finding infrequent deviations from the norm in a dataset (often a time series)
  - Recommender systems - showing other relevant items to an end user based on the activity of similar users (Amazon's product displays, Netflix recommended series for you)
  - Clustering - grouping populations within a dataset based on similar traits when "the answer" (label) is not available; an example of unsupervised learning (recommender systems can also use clustering techniques)

![img](https://s3.ap-south-1.amazonaws.com/techleer/207.jpg)

- **supervised learning**: the set of machine learning methods (algorithms) tasked with making predictions when provided *labeled data*. In other words, the algorithm is provided with the representation of each data sample. The dataset will consist of several different *features*, which should be predictors of the last column, the *labels*. Examples: predicting fraudulent activity from financial transactions, predicting whether a patient will show up to this scheduled appointment.
- **unsupervised learning**: the set of machine learning methods (algorithms) tasked with making predictions when provided *unlabeled data*. In other words, the algorithm is only provided with features, without any knowledge of what that set of features represents. In this case, the machine learning algorithm is trying to find distinct groups of data points, making the assumption that these groups have some meaning. Examples: customer segmentation for an ecommerce company, feature selection for preprocessing before running machine learning algorithm

### <u>Class 2 - Regression</u>

* **hyperparameter**: an umbrella term for the various values that can be tuned an optimized for each algorithm. These are *degrees of freedom* within each algorithm - ways of customizing an algorithm to fit the problem at hand. An example of a hyperparameter is the *learning rate*, which allows us to set how large of a step our algorithm takes when performing gradient descent.
* **linear regression**: the machine learning process by which a set of input variables (features) is mapped to a continuous output (the model). In performing a linear regression, we must assume that the relationship between our input variables can be represented by a straight line, known as the *line of best fit*. Mathematically, we find this line by minimizing the Sum of Squared Errors via Gradient Descent, but this is usually hidden from us when using machine learning implementations in Azure or Python.

![Image result for linear regression demo gif](https://cdn-images-1.medium.com/max/1600/1*2KAInY20QPhkLCJ8jWVLJw.gif)

​	There are other types of regression that we can use if  we cannot represent our data with a straight line.  These are 				called *polynomial regressions*, and could be appropriate depending on the representation of your data. 

* **logistic regression**: a machine learning algorithm used for *classification* that finds a line to separate data into two groups based on the *labels*. Thus, logistic regression is a *supervised learning* algorithm. This algorithm looks similar to a lot of other classification, but as we saw in class, different algorithms are capable of drawing different *decision boundaries*, or the line that separates our data. It gets its name from the fact that it minimizes the *log loss function* via gradient descent. Logistic regression is a very nice algorithm to use - it converges fairly quickly, and often yields very accurate results. I'd recommend trying a logistic regression as a part of any classification task. 

![Image result for logistic regression gif](https://www.cs.toronto.edu/~frossard/post/classification/anim.gif)

Notice how much slower the learning is as the epochs get higher. It starts to go up by 100 epochs with each frame because the learning is so slow. This is the nature of gradient descent. Likely in this case, the algorithm would have been accurate enough after 300 epochs or so. It almost looks like the learning rate was too low.

* **neural network**: an algorithm formed by the combination of other algorithms. This allows arbitrarily complex models to be generated as the decision boundary can be formed from many other learners. These algorithms were originally modelled on how the network of neurons in our brains make decisions. Neural networks typically refer specifically to *deep learning* algorithms that are formed by many layers of decision-making neurons. Another (more correct) term for combining weak learners into a single, stronger learner is **ensemble learning**.

* **overfitting**: this occurs when an algorithm is built too heavily on the training data and fails to generalize to other data. We typically call an algorithm overfit when it performs much better on the training data than it does on the test data. This can occur when our algorithm is too complex, when our regularization is too strong, or when we fail to properly train and validate our model.

  ![SVC_overfitting](imgs\SVC_overfitting.PNG)

* **regularization**: the process of penalizing non-predictive features and outliers to reduce the overall complexity of a model. This is commonly used to combat overfitting, though it's important to note that strong regularization can actually *cause* overfitting. There are several different types of regularization (eg. L1, L2) that can be applied depending on the nature of your machine learning task. 

* **underfitting**: this occurs when an algorithm doesn't learn enough from the training data. It doesn't perform well on either set, because it hasn't managed to capture the relationships within the dataset. This can occur when the algorithm is too simple, regularization is too weak, or our data simply doesn't predict the target variable (features aren't correlated with the answer)

  ![SVC_underfitting](imgs\SVC_underfitting.PNG)



