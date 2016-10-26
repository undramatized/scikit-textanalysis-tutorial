##Text Analytics Tutorial
###Starting a Jupyter Notebook
Start off by cloning the starter files into your local machine.
```
$ git clone https://github.com/undramatized/scikit-textanalysis-tutorial.git
```
Once you cd into the folder, start a Jupyter Session.
```
$ jupyter notebook
```
You should open a notebook in your browser. Here you should create a new iPython notebook, by selecting New > Python 2.

Once within your iPython Notebook, test that python works:
```
>>> 1+1
2
```
Cool now we can start on the Machine Learning.

###Loading the Dataset

In the following we will use the built-in dataset loader for [20 newsgroups](http://qwone.com/~jason/20Newsgroups/) from scikit-learn. 

In order to get faster execution times for this first example we will work on a partial dataset with only 4 categories out of the 20 available in the dataset:
```
>>> categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
```
We can now load the list of files matching those categories as follows:
```
>>> from sklearn.datasets import fetch_20newsgroups
>>> twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
```
The returned dataset is a `scikit-learn` “bunch”: a simple holder object with fields that can be both accessed as python `dict` keys or `object` attributes for convenience, for instance the `target_names` holds the list of the requested category names:
```
>>> twenty_train.target_names
['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
```
Here we can see more details regarding the data and its size.
```
>>> len(twenty_train.data)
>>> twenty_train.data[:1]
```
In order to train the data, every sample will need a corresponding label. In this case, the category of the data is the label that we will be working with. This can be easily retrieved as an integer using:
```
>>> twenty_train.target[:10]
```
These values will be used when we train our model, to identify the classification of each sample data.

###Extracting Features from the text data

Before we can work with text data, we need to be able to convert the text input into integers. We do this by using an approach called "Bag of Words".

**Bags of words**
 
1. assign a fixed integer `id` to each word occurring in any document of the training set (for instance by building a dictionary from words to integer indices).
2. for each document `#i`, count the number of occurrences of each word `w` and store it in `X[i, j]` as the value of feature `#j` where `j` is the index of word `w` in the dictionary.

The bags of words representation implies that `n_features` is the number of distinct words in the corpus: this number is typically larger than 100,000.

Thankfully, Scikit has a library that can handle tokenising texts so that we don't need to code it ourselves. Let's import that to use it. 
```
>>> from sklearn.feature_extraction.text import CountVectorizer
```
This library uses Scipy and Numpy, in case you were wondering why you had to download all those packages. 

We can now go ahead and initialise the bag of words counter, and vectorise our training data. 
```
>>> count_vect = CountVectorizer()
>>> X_train_counts = count_vect.fit_transform(twenty_train.data)
>>> X_train_counts.shape
(2257, 35788)
```
Now we have a dictionary of terms in our sample data. We can check the frequency index of any word using the following command. 
```
>>> count_vect.vocabulary_.get(u'apple')
```
You can check out more functions under the [CountVectorizer Docs](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer).

Although the count is a good measure of the general sentiment of a document, it is not the most accurate measure. So let's convert that count into a frequency to make it more usable. 
```
>>> from sklearn.feature_extraction.text import TfidfTransformer
>>> tfidf_transformer = TfidfTransformer()
>>> X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
>>> X_train_tfidf.shape
(2257, 35788)
```
Awesome, it's time to start training. 

###Training the classifier
SInce we have our features extracted, we can train a simple classifier. Let's start with a basic Naive Bayes classifier. 

`scikit-learn` provides a couple of variants for this particular classifier. For text analysis, a multinomial model would be the most suitable. 

If you want to understand the math, you can [read more here](http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes).

Start by importing the classifier and training it with our dataset and frequency values. 
```
>>> from sklearn.naive_bayes import MultinomialNB
>>> clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
```
That's it! We have now trained a classifier with our dataset. We can now try predicting new documents using this classifier.

###Predicting new data
Let's create some new data that we can work with:
```
>>> docs_new = ['Praise the lord', 'OpenGL on the GPU is fast', 'Rheumatology is the science of hormones']
```
This new data needs to be processed the same way that we processed the training data, so that the classifier can work with it.
```
>>> X_new_counts = count_vect.transform(docs_new)
>>> X_new_tfidf = tfidf_transformer.transform(X_new_counts)
```
Now we can use the classifier to predict the new data.
```
>>> predicted = clf.predict(X_new_tfidf)
```
Let's see what the predicted data results looks like:
```
for doc, category in zip(docs_new, predicted):
	print('%r => %s' % (doc, twenty_train.target_names[category]))
```
That's it! You can play around with different words and phrases related to the initial 4 categories we selected, and see if the classifier is able to predict accurately. 

Scikit also has tons of other [datasets](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets) and classifiers that you can play with to get a better understanding of machine learning.
