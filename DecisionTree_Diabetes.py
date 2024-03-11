#--------------------Install Libraries
#pip install scikit-learn
#pip install graphviz
#pip install pydotplus
#pip install pydotplus
#pip install graphviz

#--------------------Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import GridSearchCV

#Visualization libraries
from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image  
import pydotplus
from pydotplus import graphviz

#--------------------Pull data from computer
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("C:/Users/Andrew/WS/DecisionTree_Diabetes/diabetes.csv", header=0)
pima.columns=col_names
print(pima.head(10))

#--------------------split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

#--------------------Decision tree (First Attempt)
#Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
#Check output
print(train_test_split(X, y, test_size=0.3, random_state=1))

#Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train,)
#Predict the response for test dataset

y_pred = clf.predict(X_test)
#Check output
print(y_pred)

#Model Accuracy, how often is the classifier correct?
print("Accuracy (first attempt):",metrics.accuracy_score(y_test, y_pred))
print("Max depth is",clf.tree_.max_depth)

#Create Decision Tree Visualization
#dot_data = StringIO()
#export_graphviz(clf, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('diabetes.png')
#Image(graph.create_png())

#--------------------Second Attempt (Changing hyperparameters manually)
#Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
#Check output
#print(train_test_split(X, y, test_size=0.3, random_state=1))

#Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train,)
#Predict the response for test dataset

y_pred = clf.predict(X_test)
#Check output
#print(y_pred)

#Model Accuracy, how often is the classifier correct?
print("Accuracy (2nd Attempt):",metrics.accuracy_score(y_test, y_pred))

#Reducing the max depth to 3 increased accurracy. This means the first model was probably overfitting the training data.

#--------------------Picking hyperparameters with grid search
#Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
#Check output
#print(train_test_split(X, y, test_size=0.3, random_state=1))

#Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train,)
#Predict the response for test dataset

y_pred = clf.predict(X_test)
#Check output
#print(y_pred)

#Model Accuracy, how often is the classifier correct?
print("Accuracy (2nd Attempt):",metrics.accuracy_score(y_test, y_pred))

# Define hyperparameters to tune
param_grid = {
    'max_depth': [3, 5, 7, 9, 11, 13, 15],
    'min_samples_split': [10, 30, 50, 70, 90],
    'min_samples_leaf': [10, 30, 50, 70, 90],
    'max_features': [7],
    'criterion': ['gini', 'entropy']
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters:", grid_search.best_params_)

# Evaluate the model on test data
best_dt = grid_search.best_estimator_
accuracy = best_dt.score(X_test, y_test)
print("Grid Search Accuracy:", accuracy)

