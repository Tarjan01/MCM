from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Layer, Dropout
from keras.callbacks import Callback
from keras.optimizers import RMSprop
from keras import regularizers
from keras.utils import plot_model
from keras.layers import Input, concatenate
from keras.models import Model
from tensorflow import keras
from decorrelate import decorrelate
from sklearn.model_selection import train_test_split
import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def checkData():
    if data_frame.isnull().values.any():
        print("NaN")
    else:
        print("no NaN")
        
    nan_positions = data_frame.isnull()

    for column in nan_positions.columns:
        for index, is_nan in enumerate(nan_positions[column]):
            if is_nan:
                print(f"在列 '{column}' 的第 {index} 行找到了一个 NaN 值")

    if (data_frame.values == np.inf).any() or (data_frame.values == -np.inf).any():
        print("inf")
    else:
        print("no inf")

class CustomCallback(Callback):
    def on_train_begin(self, logs=None):
        try:
            with open('losses_train.pkl', 'rb') as f:
                self.losses_train = pickle.load(f)
        except FileNotFoundError:
            self.losses_train = []
        try:
            with open('losses_val.pkl', 'rb') as f:
                self.losses_val = pickle.load(f)
        except FileNotFoundError:
            self.losses_val = []
    def on_epoch_end(self, epoch, logs=None):
        self.losses_train.append(logs['loss'])
        self.losses_val.append(logs.get('val_loss'))
        with open('losses_train.pkl', 'wb') as f:
            pickle.dump(self.losses_train, f)
        with open('losses_val.pkl', 'wb') as f:
            pickle.dump(self.losses_val, f)
        if((epoch + 1) % mod == 0):
            fig, axs = plt.subplots(2)
            axs[0].plot(self.losses_train, label='Training Loss')
            axs[0].set_title('Training Loss')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Loss')
            axs[0].legend()

            axs[1].plot(self.losses_val, label='Validation Loss', color='orange')
            axs[1].set_title('Validation Loss')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('Loss')
            axs[1].legend()

            plt.tight_layout()
            plt.show()

        if (epoch + 1) % 10 == 0:  
            print('After epoch {}, training loss is {}, validation loss is {}\n'.format(len(self.losses_train), logs['loss'], logs.get('val_loss')))

class MyGenerator(keras.utils.Sequence):
    def __init__(self, X_data, y_data, st):
        self.X, self.y = X_data, y_data
        self.st = st

    def __len__(self):
        return len(self.st) - 1

    def __getitem__(self, idx):
        batch_x = self.X[self.st[idx]:self.st[idx + 1]]
        batch_y = self.y[self.st[idx]:self.st[idx + 1]]
        
        return batch_x, batch_y

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")        
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(Attention, self).get_config()

def firstTrain():
    main_input = Input(shape=(11, 1), name='main_input')
    group_input = Input(shape=(1,), name='group_input')
    
    lstm_out = LSTM(unit, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(main_input)
    lstm_out = Dropout(0.5)(lstm_out)
    lstm_out = Attention()(lstm_out)

    main_output = Dense(1, kernel_regularizer=regularizers.l2(0.01), name='main_output')(lstm_out)

    model = Model(inputs=[main_input], outputs=[main_output])

    model.compile(optimizer=RMSprop(learning_rate=my_learningRate), loss='mse')
    print("posfit")
    
    training_generator = MyGenerator(X_train, y_train, st_train)
    validation_generator = MyGenerator(X_val, y_val, st_val)    
    
    model.fit(training_generator, epochs=my_epoch, callbacks=[custom_callback], verbose=0, validation_data=validation_generator)
    model.save('my_model.h5', save_format='h5', include_optimizer=True)

def calcAq():
    model = load_model('my_model.h5', custom_objects={'Attention': Attention})
    y_pred = model.predict(X_all)
    data = pd.read_csv('./assets/doc/data3.csv', encoding='utf-8')
    
    file_path = "./assets/doc/output.txt"
    score = pd.read_csv(file_path, header=None, names=["score1", "score2"])
    score = score.score2 - score.score1
    
    dec = decorrelate(score, data.serve)
    
    y_pred = np.squeeze(y_pred)
    y_pred=dec.inverse(y_pred,data.serve)
    
    winner = pd.read_csv('./assets/doc/winner.csv', encoding='utf-8')
    winner_pred = [0] * y_pred.shape[0]
    print(winner.shape[0], y_pred.shape[0])
    # exit()
    for i in range(y_pred.shape[0]):
        if(y_pred[i] > 0):
            winner_pred[i] = 2
        else:
            winner_pred[i] = 1
    
    valPredcorrect = 0
    for i in range(indices_val.shape[0]):
        if(winner_pred[indices_val[i][0]] == winner.iloc[indices_val[i][0]].values[0]):
            valPredcorrect += 1
    
    testPredcorrect = 0
    for i in range(indices_test.shape[0]):
        if(winner_pred[indices_test[i][0]] == winner.iloc[indices_test[i][0]].values[0]):
            testPredcorrect += 1
    
    print(f"valAccuracy: {valPredcorrect / indices_val.shape[0]}")
    print(f"testAccuracy: {testPredcorrect / indices_test.shape[0]}")
def calcAq2():
    model = load_model('my_model.h5', custom_objects={'Attention': Attention})
    y_pred = model.predict(X_all)
    data = pd.read_csv('./assets/doc/data3.csv', encoding='utf-8')
    
    file_path = "./assets/doc/output.txt"
    score = pd.read_csv(file_path, header=None, names=["score1", "score2"])
    score = score.score2 - score.score1
    
    dec = decorrelate(score, data.serve)
    
    y_pred = np.squeeze(y_pred)
    y_pred=dec.inverse(y_pred,data.serve)
    
    winner = pd.read_csv('./assets/doc/winner.csv', encoding='utf-8')
    winner_pred = [0] * y_pred.shape[0]
    print(winner.shape[0], y_pred.shape[0])
    # exit()
    for i in range(y_pred.shape[0]):
        if(y_pred[i] > 0):
            winner_pred[i] = 2
        else:
            winner_pred[i] = 1
    
    for i in range(len(st_test) - 1):
        testPredcorrect = 0
        for j in range(st_test[i], st_test[i + 1], 1):
            if(winner_pred[j] == winner.iloc[j].values[0]):
                testPredcorrect += 1
        print(f"testAccuracy: {testPredcorrect / (st_test[i + 1] - st_test[i])}")
        

def continueTrain():
    model = load_model('my_model.h5', custom_objects={'Attention': Attention})
    training_generator = MyGenerator(X_train, y_train, st_train)
    validation_generator = MyGenerator(X_val, y_val, st_val)
    model.fit(training_generator, epochs=my_epoch, callbacks=[custom_callback], verbose=0, validation_data=validation_generator)
    model.save('my_model.h5', save_format='h5', include_optimizer=True)
    
def changeLearningRateandContinueTrain():
    model = load_model('my_model.h5', custom_objects={'Attention': Attention})
    model.compile(optimizer=RMSprop(learning_rate=my_learningRate), loss='mse')
    training_generator = MyGenerator(X_train, y_train, st_train)
    validation_generator = MyGenerator(X_val, y_val, st_val)
    
    model.fit(training_generator, epochs=my_epoch, callbacks=[custom_callback], verbose=0, validation_data=validation_generator)
    model.save('my_model.h5', save_format='h5', include_optimizer=True)

def showFig():
    model = Sequential()
    model.add(LSTM(unit, return_sequences=True, input_shape=(10, 1)))
    kernel_regularizer=regularizers.l2(1)
    model.add(Dropout(0.5))
    model.add(Attention())
    model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    
    newLearningRate = my_learningRate
    optimizer = RMSprop(learning_rate=newLearningRate)

    model.compile(optimizer=optimizer, loss='mse') 
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96, layer_range=None)

def load_data(num):
    now = 0
    i = 0
    while now < num:
        i += 1
        if(data_frame.iloc[i][0] != data_frame.iloc[i-1][0]):
            now += 1
    j = i
    while j < data_frame.shape[0] and data_frame.iloc[j][0] == data_frame.iloc[i][0]:
        j += 1
    return data_frame.iloc[i:j, 2:13], data_frame.iloc[i:j, 1:2]
    

if __name__ == "__main__":
    print("pos0")
    data_frame = pd.read_csv('./assets/doc/inputdata_updated.csv', encoding='utf-8',)
    
    X_all = []
    X_train = []
    st_train = []
    X_val = []
    st_val = []
    X_test = []
    st_test = []
    y_train = []
    y_val = []
    y_test = []
    

    indices_train = []
    indices_val = []
    indices_test = []
    
    tot = 0
    
    train_size = 31*0.6
    val_size = (31 - train_size) * 0.5
    
    for i in range(31):
        ## 0-30
        X, y = load_data(i)
        n = X.shape[0]
        indices = range(tot, tot + n)
        tot += n
        X_all.append(X)
        if(i == train_size):
            st_train.append(sum([x.shape[0] for x in X_train]))
        if(i == train_size + val_size):
            st_val.append(sum([x.shape[0] for x in X_val]))
        if(i < train_size):
            st_train.append(sum([x.shape[0] for x in X_train]))
            X_train.append(X)
            y_train.append(y)
            indices_train.extend(indices)
        elif (i < train_size + val_size):
            st_val.append(sum([x.shape[0] for x in X_val]))
            X_val.append(X)
            y_val.append(y)
            indices_val.extend(indices)
        else:
            st_test.append(sum([x.shape[0] for x in X_test]))
            X_test.append(X)
            y_test.append(y)
            indices_test.extend(indices)
    st_test.append(sum([x.shape[0] for x in X_test]))
    
    X_all = np.concatenate(X_all, axis=0)
    X_train = np.concatenate(X_train, axis=0)
    X_val = np.concatenate(X_val, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    indices_train = np.array(indices_train).reshape(-1, 1)
    indices_val = np.array(indices_val).reshape(-1, 1)
    indices_test = np.array(indices_test).reshape(-1, 1)
    
    print("pos1")
    unit = 32
    my_epoch = 100
    mod = 100
    my_learningRate = 0.0001
    custom_callback = CustomCallback()
    # continueTrain()
    # changeLearningRateandContinueTrain()
    # firstTrain()
    # showFig()
    # calcAq()
    calcAq2()