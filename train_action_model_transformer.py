
import pandas as pd
from enum import Enum
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential,Model 
from keras.layers import Flatten,MaxPooling1D,Conv1D,SimpleRNN,Multiply,Activation,BatchNormalization
from keras.layers import Dense, Dropout,LSTM,GRU,Bidirectional,GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import concatenate,Input, SpatialDropout1D,Embedding,MultiHeadAttention,LayerNormalization
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam,SGD
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import itertools
from sklearn.metrics import confusion_matrix
import time
import tensorflow as tf
from keras.utils.vis_utils import plot_model
#自定义注意力层
from keras import initializers, constraints,activations,regularizers
from keras import backend as K
from keras.layers import Layer
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import seaborn as sns

np.random.seed(10)

model_arc = r'C:\Users\Citydo\Documents\yyk\zju\mypaper\code\3Daction_videowise_action_recognition\Action\training\figure\model'
raw_data = pd.read_csv(r'C:\Users\Citydo\Documents\yyk\zju\mypaper\code\3Daction_videowise_action_recognition\Action\training\DLC_3D.csv', header=0)
weight_path = r'C:\Users\Citydo\Documents\yyk\zju\mypaper\code\3Daction_videowise_action_recognition\Action\training\weights'
figure_acc = r'C:\Users\Citydo\Documents\yyk\zju\mypaper\code\3Daction_videowise_action_recognition\Action\training\figure\acc'
figure_cm = r'C:\Users\Citydo\Documents\yyk\zju\mypaper\code\3Daction_videowise_action_recognition\Action\training\figure\cm'

# MODEL_TYPE = 'RNN'
# MODEL_TYPE = 'LSTM'
# MODEL_TYPE = 'GRU'
# MODEL_TYPE = 'BiGRU'
# MODEL_TYPE = 'BiLSTM'
# MODEL_TYPE = 'MLP'
# MODEL_TYPE = '1DCNN'
# MODEL_TYPE = '1DCNN-LSTM'
# MODEL_TYPE = 'Attention'
# MODEL_TYPE = 'BiGRU-Attention'

# SequenceLength = 5
# SequenceLength = 10
# SequenceLength = 15
# SequenceLength = 30

epochs = 5
KPSIZE = 18
num_classes = 5
test_size = 0.4
first_RNN_layer = GRU
first_RNN_layer_channel = 64
dropout_rate =0.4
RNN_BATCH_SIZE = 64
MAX_FRAMES = 15300
LR = 0.001
patience = 50
dense_channel_1 = 32
dense_channel_2 = 16

rest = 3270
stand = 2160
arrangement = 4800
sniff = 2850
walk = 2220


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = Sequential([layers.Dense(dense_dim, activation="relu"),layers.Dense(embed_dim),] )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim, })
        return config

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,})
        return config

def action_report(model,X_test,Y_test_original):
    #预测概率
    prob=model.predict(X_test) 
    #预测类别
    pred=np.argmax(prob,axis=1)
    #计算混淆矩阵的各项指标
    print(classification_report(Y_test_original.argmax(axis=1), pred))

def plot_loss(history,file_name,figure_acc):
    plt.plot(history.history['loss'], 'g', label='train loss')
    plt.plot(history.history['accuracy'], 'r', label='train acc')
    # plt.title('train loss')
    plt.ylabel('acc-loss')
    plt.xlabel('epoch')
    plt.legend(loc="upper right")# 画出损失函数曲线
    plt.plot(history.history['val_loss'],   'b', label='val loss')
    plt.plot(history.history['val_accuracy'], 'k', label='val acc')
    # plt.title('val loss')
    plt.grid(True)
    plt.ylabel('acc-loss')
    plt.xlabel('epoch')
    plt.legend(loc="upper right")
    # plt.show()
    plt.savefig(os.path.join(figure_acc,r'{}.png'.format(file_name)))

class Attention(Layer):
    #返回值：返回的不是attention权重，而是每个timestep乘以权重后相加得到的向量。
    #输入:输入是rnn的timesteps，也是最长输入序列的长度
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
 
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
 
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) == 3
 
        self.W = self.add_weight(shape=(input_shape[-1],),initializer=self.init,name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
 
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),initializer='zero', name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True
 
    def compute_mask(self, input, input_mask=None):
        return None     ## 后面的层不需要mask了，所以这里可以直接返回none
 
    def call(self, x, mask=None):
        features_dim = self.features_dim    ## 这里应该是 step_dim是我们指定的参数，它等于input_shape[1],也就是rnn的timesteps
        step_dim = self.step_dim
        
        # 输入和参数分别reshape再点乘后，tensor.shape变成了(batch_size*timesteps, 1),之后每个batch要分开进行归一化
         # 所以应该有 eij = K.reshape(..., (-1, timesteps))
 
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b        
        eij = K.tanh(eij)    #RNN一般默认激活函数为tanh, 对attention来说激活函数差别不大，因为要做softmax
        a = K.exp(eij)
        if mask is not None:    ## 如果前面的层有mask，那么后面这些被mask掉的timestep肯定是不能参与计算输出的，也就是将他们attention权重设为0
            a *= K.cast(mask, K.floatx())   ## cast是做类型转换，keras计算时会检查类型，可能是因为用gpu的原因
 
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)      # a = K.expand_dims(a, axis=-1) , axis默认为-1， 表示在最后扩充一个维度。比如shape = (3,)变成 (3, 1)
        ## 此时a.shape = (batch_size, timesteps, 1), x.shape = (batch_size, timesteps, units)
        weighted_input = x * a    
        # weighted_input的shape为 (batch_size, timesteps, units), 每个timestep的输出向量已经乘上了该timestep的权重
        # weighted_input在axis=1上取和，返回值的shape为 (batch_size, 1, units)
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):    ## 返回的结果是c，其shape为 (batch_size, units)
        return input_shape[0],  self.features_dim

# Callback class to visialize training progress
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0,fontsize = 18)
    plt.yticks(tick_marks, classes,fontsize = 18)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),fontsize = 16,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.1,bottom=0.08)
    plt.ylabel('True label',fontsize = 20)
    plt.xlabel('Predicted label',fontsize = 20)

# confusion matrix
def plot_cm(Y_test,Y_pred,file_name):
    cfm = confusion_matrix(np.argmax(Y_test,axis=1), np.argmax(Y_pred, axis=1))
    np.set_printoptions(precision=2)
    plt.figure(figsize=(16, 12), dpi=100)
    class_names = ['rest','stand','arrange','sniff','walk']
    plot_confusion_matrix(cfm, classes=class_names, normalize = True,title='Rat Action Confusion Matrix')
    # plt.show()
    plt.savefig(os.path.join(figure_cm,r'{}.png'.format(file_name)))

def find_newest_file(path_file):
    lists = os.listdir(path_file)
    lists.sort(key=lambda fn: os.path.getmtime(path_file +'\\'+fn))
    # print('weight-lists:',lists)
    file_newest = os.path.join(path_file,lists[-1])
    return file_newest

#build new lstm-model 2022-11-18
def build_model(MODEL_TYPE,SequenceLength):
    if MODEL_TYPE == 'RNN':
        model = Sequential()
        model.add(SimpleRNN(dense_channel_1, input_shape= (SequenceLength,KPSIZE), return_sequences=False))
        model.add(Dropout(dropout_rate))  
        # model.add(BatchNormalization())
        model.add(Dense(num_classes, activation="softmax"))
        model.summary()
    elif MODEL_TYPE=='MLP':
        model = Sequential()
        model.add(Input(shape=(SequenceLength,KPSIZE)))
        model.add(Flatten())
        model.add(Dense(dense_channel_1, activation="relu"))  
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation="softmax"))
        model.summary()
    elif MODEL_TYPE=='1DCNN':        #一维卷积
        model = Sequential()
        model.add(Conv1D(input_shape = (SequenceLength,KPSIZE),filters=32, kernel_size=3, padding="same",activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(dense_channel_1, activation="relu"))
        model.add(Dropout(dropout_rate))   
        model.add(Dense(num_classes, activation="softmax"))
        model.summary()
    elif MODEL_TYPE=='1DCNN-LSTM':
        model = Sequential()  
        model.add(Conv1D(input_shape = (SequenceLength,KPSIZE),filters=32, kernel_size=3, padding="same",activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(dense_channel_1))
        model.add(Dropout(dropout_rate))   
        model.add(Dense(num_classes, activation="softmax"))
        model.summary()
    elif MODEL_TYPE=='BiLSTM':
        inp = Input(shape=(SequenceLength,KPSIZE))
        x = Bidirectional(LSTM(dense_channel_1, return_sequences=True))(inp)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])
        # forward_layer = LSTM(64, return_sequences=True)
        # backward_layer = LSTM(64, activation='relu', return_sequences=True,go_backwards=True)
        # model.add(Bidirectional(forward_layer, backward_layer=backward_layer,input_shape=(SequenceLength,KPSIZE)))
        x = Dense(dense_channel_2, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        outp = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=inp, outputs=outp)
        model.summary()
    elif MODEL_TYPE == 'BiGRU':
        inp = Input(shape=(SequenceLength,KPSIZE))
        # inp = Dropout(dropout_rate)(inp)
        x = Bidirectional(GRU(dense_channel_1, return_sequences=True))(inp)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])
        x = Dense(dense_channel_2, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        outp = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=inp, outputs=outp)
        model.summary()
    elif MODEL_TYPE == 'LSTM':
        model = Sequential()
        model.add(LSTM(dense_channel_1, input_shape= (SequenceLength,KPSIZE), return_sequences=False))
        model.add(Dense(dense_channel_2, activation = 'relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()
    elif MODEL_TYPE == 'GRU':
        model = Sequential()
        model.add(GRU(dense_channel_1, input_shape= (SequenceLength,KPSIZE), return_sequences=False))
        # model.add(Dropout(dropout_rate))
        model.add(Dense(dense_channel_2, activation = 'relu'))
        model.add(Dropout(dropout_rate))
        # model.add(Dense(dense_channel_2, activation = 'relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()
    elif MODEL_TYPE=='Attention':
        inputs = Input(name='inputs',shape=(SequenceLength,KPSIZE), dtype='float64')
        attention_probs = Dense(KPSIZE, activation='softmax', name='attention_vec')(inputs)
        attention_mul =  Multiply()([inputs, attention_probs])
        mlp = Dense(dense_channel_1)(attention_mul) #原始的全连接
        fla=Flatten()(mlp)
        output = Dense(num_classes, activation='softmax')(fla)
        model = Model(inputs=[inputs], outputs=output)
        model.summary()
    elif MODEL_TYPE == 'Attention-BiGRU':
        inputs = Input(name='inputs',shape=(SequenceLength,KPSIZE), dtype='float64')
        attention_probs = Dense(KPSIZE, activation='softmax', name='attention_vec')(inputs)
        attention_mul =  Multiply()([inputs, attention_probs])
        mlp = Dense(dense_channel_1,activation='relu')(attention_mul) #原始的全连接
        x = Bidirectional(GRU(dense_channel_2, return_sequences=True))(mlp)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        dense = Dense(dense_channel_2, activation="relu")(conc)
        outp = Dense(num_classes, activation="softmax")(dense)
        model = Model(inputs=inputs, outputs=outp)
        model.summary()
    elif MODEL_TYPE =='BiGRU-Attention':
        inputs = Input(shape=(SequenceLength,KPSIZE))
        x = GRU(dense_channel_1, return_sequences=True)(inputs)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
        x = Attention(SequenceLength)(x)
        x = Dense(dense_channel_2, activation="relu")(x)
        outp = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outp)
    elif MODEL_TYPE=='Transformer':
        inputs = Input(name='inputs',shape=(SequenceLength,KPSIZE), dtype='float64')
        # x = Embedding(top_words, input_length=max_words, output_dim=embed_dim, mask_zero=True)(inputs)
        x = TransformerEncoder(KPSIZE, 512, 4)(inputs)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs, outputs)
        model.summary()
    elif MODEL_TYPE =='Transformer+BiGRU':
        inputs = Input(name='inputs',shape=(SequenceLength,KPSIZE), dtype='float64')
        # x = Embedding(top_words, input_length=max_words, output_dim=embed_dim, mask_zero=True)(inputs)
        x = TransformerEncoder(KPSIZE, 512, 4)(inputs)
        # x = GlobalMaxPooling1D()(x)
        # x = Dropout(dropout_rate)(x)
        x = Bidirectional(GRU(dense_channel_1, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])
        x = Dropout(dropout_rate)(x)
        x = Dense(dense_channel_2, activation="relu")(x)
        # x = Dropout(dropout_rate)(x)
        outp = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outp)
        model.summary()
    elif MODEL_TYPE =='Transformer+GRU':
        inputs = Input(name='inputs',shape=(SequenceLength,KPSIZE), dtype='float64')
        # x = TransformerEncoder(KPSIZE, dense_channel_1, 6)(inputs)
        # x= PositionalEmbedding(sequence_length=SequenceLength, input_dim=MAX_FRAMES, output_dim=KPSIZE)(inputs)
        x = TransformerEncoder(KPSIZE, 512, 4)(inputs)
        x = GRU(dense_channel_1, return_sequences=False)(x)
        x = Dense(dense_channel_2, activation="relu")(x)
        # x = Dropout(dropout_rate)(x)
        outp = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outp)
        model.summary()
    elif MODEL_TYPE =='Transformer+RNN':
        inputs = Input(name='inputs',shape=(SequenceLength,KPSIZE), dtype='float64')
        # x = TransformerEncoder(KPSIZE, dense_channel_1, 6)(inputs)
        # x= PositionalEmbedding(sequence_length=SequenceLength, input_dim=MAX_FRAMES, output_dim=KPSIZE)(inputs)
        x = TransformerEncoder(KPSIZE, 512, 4)(inputs)
        x = SimpleRNN(dense_channel_1, return_sequences=False)(x)
        x = Dense(dense_channel_2, activation="relu")(x)
        # x = Dropout(dropout_rate)(x)
        outp = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outp)
        model.summary()
    elif MODEL_TYPE=='Transformer+MLP':
        inputs = Input(name='inputs',shape=(SequenceLength,KPSIZE), dtype='float64')
        x = TransformerEncoder(KPSIZE, 512, 4)(inputs)
        x = Flatten()(x)
        x = Dense(dense_channel_1, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        outp = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outp)
        model.summary()
    elif MODEL_TYPE=='Transformer+1DCNN':
        inputs = Input(name='inputs',shape=(SequenceLength,KPSIZE), dtype='float64')
        x = TransformerEncoder(KPSIZE, 512, 4)(inputs)
        x = Conv1D(input_shape = (SequenceLength,KPSIZE),filters=32, kernel_size=3, padding="same",activation="relu")(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(dense_channel_1, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        outp = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outp)
        model.summary()
    elif MODEL_TYPE=='Transformer+LSTM':
        inputs = Input(name='inputs',shape=(SequenceLength,KPSIZE), dtype='float64')
        x = TransformerEncoder(KPSIZE, 512, 4)(inputs)
        x = LSTM(dense_channel_1, input_shape= (SequenceLength,KPSIZE), return_sequences=False)(x)
        x = Dense(dense_channel_1, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        outp = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outp)
        model.summary()
    elif MODEL_TYPE =='Transformer+BiLSTM':
        inputs = Input(name='inputs',shape=(SequenceLength,KPSIZE), dtype='float64')
        # x = Embedding(top_words, input_length=max_words, output_dim=embed_dim, mask_zero=True)(inputs)
        x = TransformerEncoder(KPSIZE, 512, 4)(inputs)
        # x = GlobalMaxPooling1D()(x)
        # x = Dropout(dropout_rate)(x)
        x = Bidirectional(LSTM(dense_channel_1, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])
        x = Dropout(dropout_rate)(x)
        x = Dense(dense_channel_2, activation="relu")(x)
        # x = Dropout(dropout_rate)(x)
        outp = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outp)
        model.summary()
    return model

# load data training
def process_data(raw_data,SequenceLength):
    SequenceLength = int(SequenceLength)
    dataset = raw_data.values
    # raw_data['classes'].value_counts().plot(kind='bar')
    print(raw_data['classes'].value_counts())
    X = dataset[:, 0:KPSIZE].astype(float)
    Y = dataset[:, KPSIZE]
    blocks = int(len(X) / int(SequenceLength))
    X = np.array(np.split(X, blocks))

    # 将类别编码为数字,共1350
    encoder_Y = [0]*int(rest/SequenceLength) + [1]*int(stand/SequenceLength) + [2]*int(arrangement/SequenceLength) + \
        [3]*int(sniff/SequenceLength) + [4]*int(walk/SequenceLength)
    dummy_Y = np_utils.to_categorical(encoder_Y) # one hot 编码

    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=test_size, random_state=9)
    Y_test_original = Y_test.copy()

    return X_train, X_test, Y_train, Y_test,Y_test_original

def train(MODEL_TYPE,SequenceLength,epochs,filepath):
    X_train, X_test, Y_train, Y_test,Y_test_original =  process_data(raw_data,SequenceLength)
    
    model = build_model(MODEL_TYPE,SequenceLength)
    plot_model(model, to_file=os.path.join(model_arc,r'{}.png'.format(MODEL_TYPE)), show_shapes=True, show_layer_names=True )
    model.compile(loss='categorical_crossentropy', optimizer=Adam(LR), metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath,  monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    history = model.fit(X_train, Y_train, batch_size=128, epochs=epochs, verbose=1, shuffle=True,validation_data=(X_test, Y_test), callbacks=[checkpoint])
    newest_weight = find_newest_file(weight_path)

    # evaluate and draw confusion matrix
    file_name = newest_weight.split("\\")[-1].split('.hdf5')[0]
    print(file_name)
    if MODEL_TYPE == 'Transformer':
        model = load_model(newest_weight, custom_objects={'TransformerEncoder': TransformerEncoder})
    elif MODEL_TYPE == 'Transformer+BiGRU':
        model = load_model(newest_weight, custom_objects={'TransformerEncoder': TransformerEncoder})
    elif MODEL_TYPE == 'Transformer+GRU':
        model = load_model(newest_weight, custom_objects={'TransformerEncoder': TransformerEncoder})
    elif MODEL_TYPE == 'Transformer+RNN':
        model = load_model(newest_weight, custom_objects={'TransformerEncoder': TransformerEncoder})
    elif MODEL_TYPE == 'Transformer+MLP':
        model = load_model(newest_weight, custom_objects={'TransformerEncoder': TransformerEncoder})
    elif MODEL_TYPE == 'Transformer+1DCNN':
        model = load_model(newest_weight, custom_objects={'TransformerEncoder': TransformerEncoder})
    elif MODEL_TYPE=='Transformer+LSTM':
        model = load_model(newest_weight, custom_objects={'TransformerEncoder': TransformerEncoder})
    elif MODEL_TYPE=='Transformer+BiLSTM':
        model = load_model(newest_weight, custom_objects={'TransformerEncoder': TransformerEncoder})
    else:
        model = load_model(newest_weight)
    plot_loss(history,file_name,figure_acc)
    action_report(model=model,X_test=X_test,Y_test_original=Y_test_original)
    Y_pred = model.predict(X_test)
    plot_cm(Y_test,Y_pred,file_name)
    t1 = time.time()
    score, accuracy = model.evaluate(X_test,Y_test,batch_size=64)
    print('time-inference:',time.time()-t1)
    print('Test accuracy:{:.5}'.format(accuracy))

test_size = 0.1
dropout_rate =0.3
LR = 0.001

if __name__ == '__main__':
    # ['RNN','LSTM','GRU','BiGRU','BiLSTM','MLP','1DCNN','1DCNN-LSTM','Attention','Attention-BiGRU',Transformer,
    # Transformer+BiGRU,Transformer+GRU,Transformer+MLP,Transformer+1DCNN,Transformer+RNN,Transformer+LSTM,Transformer+BiLSTM]


    filepath=r'C:\Users\Citydo\Documents\yyk\zju\mypaper\code\3Daction_videowise_action_recognition\Action\training\weights\Transformer-L1-S30-{epoch:02d}-{val_accuracy:.5f}.hdf5'

    train(MODEL_TYPE = 'Transformer+BiGRU', SequenceLength=30, epochs=200, filepath = filepath)