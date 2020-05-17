import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("/home/sadabrata/Datasets/Churn_Modelling.csv")
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

geography = pd.get_dummies(X["Geography"], drop_first = True)
gender = pd.get_dummies(X["Gender"], drop_first = True)

X = pd.concat([X, geography, gender], axis = 1)
X.drop(["Geography", "Gender"], axis =1, inplace = True)
#X.columns

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(units = 15, kernel_initializer = 'he_uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 20, kernel_initializer = 'he_uniform', activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units = 2, activation='softmax'))
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

model = classifier.fit(X_train, y_train, validation_split = 0.30, batch_size = 10, nb_epoch = 100)

print(model.history.keys())

plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

scores = classifier.evaluate(X_test, y_test)
print("Accuracy = ", scores[1])
