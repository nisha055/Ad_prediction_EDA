# --------------------------------------------Import modules ----------------------------
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score

import pickle

# --------------------------------------------Read CSV with pandas ----------------------------
data = pd.read_csv(r'advertising_updated.csv')

# -------------------------------------------- ANN using keras and Tensorflow ----------------------------

# Convert the data into an array
dataset = data.values
# Get all of the rows for all columns except last
X = dataset[:, 0:12]
# Get  all of the rows for last column
y = dataset[:, 12]

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scale, y, test_size=0.2, random_state=4)

# Building keras model
# Sequential  - creating model layer by layer
model = Sequential([
    # Dense layer - fully connected neural network
    # First hidden layer with 12 neurons and 12 dimensions in input
    Dense(12, activation='relu', input_shape=(12,)),
    # 2nd hidden layer with 15 neurons
    # Relu to avoid vanishing gradient problem
    Dense(15, activation='relu'),
    # Output layer and since o/p is binary hence 1 neuron
    # Sigmoid used for classification problems
    Dense(1, activation='sigmoid')
])

# Compiling the model using sgd as optimizer function and binary_crossentropy as loss function
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fitting the model
ANN_model = model.fit(X_train, y_train,
                      batch_size=57, epochs=1000, validation_split=0.2)

# Make a prediction & print the actual values
pred = model.predict(X_test)
pred = [1 if y >= 0.5 else 0 for y in pred]
print()
print('\n Artificial Neural Network Accuracy: ', accuracy_score(y_test, pred))
print('\n Confusion Matrix: \n', confusion_matrix(y_test, pred))

# -------------------------------------------- TRAINING AND TEST DATA FOR OTHER MODELS ----------------------------
X = data[['Daily_Time_Spent_on_Site', 'Age', 'Area_Income', 'Daily_Internet_Usage',
          'Male', 'City_Codes', 'Country_Codes', 'Month', 'Day_of_the_month', 'Day_of_the_week', 'Hour']]
y = data['Clicked_on_Ad']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# -------------------------------------------- FEW OTHER MODELS ----------------------------
# --------------------------------------------Logistic Regression ----------------------------
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(X_train, y_train)
predictions_LR = LR_model.predict(X_test)

print('\n Logistic regression accuracy:',
      accuracy_score(predictions_LR, y_test))
print('\n Confusion matrix:')
print(confusion_matrix(y_test, predictions_LR))

# Save model
with open('LR_model.pickle', 'wb') as f:
    pickle.dump(LR_model, f)


# -------------------------------------------- Decision Trees ----------------------------
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train, y_train)
predictions_DT = DT_model.predict(X_test)

print('\n Decision tree accuracy:', accuracy_score(predictions_DT, y_test))
print('\n Confusion matrix:')
print(confusion_matrix(y_test, predictions_DT))

# Save model
with open('DT_model.pickle', 'wb') as f:
    pickle.dump(DT_model, f)

# -------------------------------------------- Naive Bayes ----------------------------
NB_model = GaussianNB()
NB_model.fit(X_train, y_train)
predictions_NB = NB_model.predict(X_test)

print('\n Naive Bayes accuracy:', accuracy_score(predictions_NB, y_test))
print('\n Confusion matrix:')
print(confusion_matrix(y_test, predictions_NB))

# Save model
with open('NB_model.pickle', 'wb') as f:
    pickle.dump(NB_model, f)
