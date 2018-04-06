'''

>>> python3 train.py filepath epochs

And the model will start training on the given dataset.


'''




from keras.layers import Dense,LSTM,Embedding
from keras.utils import to_categorical
from keras.models import Sequential
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.cross_validation import train_test_split
import argparse
import matplotlib.pyplot as plt


parser=argparse.ArgumentParser()
parser.add_argument('filepath',type=str)
parser.add_argument('epochs',type=int)
args=parser.parse_args()


 

file=open(filepath)
string=file.read()
words=string.split()

sentences=[words[i:i+5] for i in range(len(string)-5)]
targets=[words[i+5] for i in range(len(string)-5)]


vocab=list(set(words))

words_to_code=dict((i,j) for j,i in enumerate(vocab))
code_to_words=dict((i,j) for i,j in enumerate(vocab))
newtargets=[words_to_code[w] for w in targets]
for i in range(len(sentences)):
    for j in range(5):
        sentences[i][j]=words_to_code[sentences[i][j]]
sentences=np.array(sentences)



newtargets=to_categorical(newtargets,num_classes=len(vocab))
newtargets=np.array(newtargets)

X_train,X_test,Y_train,Y_test=train_test_split(sentences,newtargets)

model=Sequential()
model.add(Embedding(len(vocab),40,input_length=5))
model.add(LSTM(40,dropout=0.3,input_shape=(5,40),return_sequences=True))
model.add(LSTM(40,dropout=0.3))
model.add(Dense(len(vocab),activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

history=model.fit(X_train,Y_train,epochs=epochs,validation_data=(X_test,Y_test))


plt.plot(history.history['loss'])
plt.show()