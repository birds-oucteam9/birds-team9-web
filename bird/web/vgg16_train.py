#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget https://dwz.cn/ijPVPQhz')
get_ipython().system('sudo apt-get install rar unrar')
get_ipython().system('unrar x ijPVPQhz')
get_ipython().system('cp -r AI研习社_鸟类识别比赛数据集/* ./')


# In[2]:


# 将kaggle的数据集直接下载到codelab中
get_ipython().system('pip install -U -q kaggle --upgrade')
get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('echo \'{"username":"codingchaozhang","key":"4f6ee69ad1970ff0c67499e6defcefb7"}\' > ~/.kaggle/kaggle.json')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')
get_ipython().system('kaggle competitions download -c birds-classification')


# In[1]:


import os

train_set_dir = "train_set/"
val_set_dir = "val_set/"
test_set_dir = "test_set/"

print(len(os.listdir(train_set_dir)))
print(len(os.listdir(test_set_dir)))
print(len(os.listdir(test_set_dir)))


# ### 探索数据

# In[2]:


import os

bird_dir = "/content/"
x_train_path = os.path.join(bird_dir,"train_set")
x_test_path = os.path.join(bird_dir,"test_set")
x_valid_path = os.path.join(bird_dir,"val_set")

y_train_path = os.path.join(bird_dir,"train_pname_to_index.csv")
y_valid_path = os.path.join(bird_dir,"val_pname_to_index.csv")


# In[3]:


import pandas as pd

y_train = pd.read_csv(y_train_path,skiprows=0)
y_valid = pd.read_csv(y_valid_path,skiprows=0)


# In[4]:


y_train.head()


# In[5]:


y_valid.head()


# In[6]:


x_train_img_path = y_train["img_path"]
y_train = y_train["label"] - 1
x_valid_img_path = y_valid["img_path"]
y_valid = y_valid["label"] -1

print(x_train_img_path[:5])
print(y_train[:5])

print(x_valid_img_path[:5])
print(y_valid[:5])


# In[7]:


# 定义读取图片函数
import cv2
import numpy as np

def get_img(file_path,img_rows,img_cols):
  
    img = cv2.imread(file_path)
    img = cv2.resize(img,(img_rows,img_cols))
    if img.shape[2] == 1:
      img = np.dstack([img,img,img])
    else:
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    
    return img


# In[8]:


# 加载训练集
x_train = []
for img_name in x_train_img_path:
    img = get_img(os.path.join(x_train_path,img_name),224,224)
    x_train.append(img)

x_train = np.array(x_train,np.float32)


# In[9]:


# 加载验证集
x_valid = []
for img_name in x_valid_img_path:
    img = get_img(os.path.join(x_valid_path,img_name),224,224)
    x_valid.append(img)

x_valid = np.array(x_valid,np.float32)


# In[10]:


# 加载预测集
import re

x_test_img_path = os.listdir(x_test_path)
x_test_img_path = sorted(x_test_img_path,key = lambda i:int(re.match(r"(\d+)",i).group()))


x_test = []
for img_name in x_test_img_path:
    img = get_img(os.path.join(x_test_path,img_name),224,224)
    x_test.append(img)

x_test = np.array(x_test,np.float32)


# In[11]:


print(x_train.shape)
print(y_train.shape)

print(x_valid.shape)
print(y_valid.shape)

# print(x_test.shape)


# In[12]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.imshow(x_train[0]/255)
print(y_train[0])


# In[13]:


X_train = np.concatenate((x_train,x_valid),axis=0)
Y_train = np.concatenate((y_train,y_valid),axis=0)

print(X_train.shape)
print(Y_train.shape)


# print(x_test.shape)


# In[14]:


sum = np.unique(y_train)
n_classes = len(sum)


# In[15]:


# 直方图来显示图像训练集的各个类别的分别情况
def plot_y_train_hist():
  fig = plt.figure(figsize=(15,5))
  ax = fig.add_subplot(1,1,1)
  hist = ax.hist(Y_train,bins=n_classes)
  ax.set_title("the frequentcy of each category sign")
  ax.set_xlabel("bird")
  ax.set_ylabel("frequency")
  plt.show()
  return hist

hist = plot_y_train_hist()


# In[26]:


# 划分数据集
from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid = train_test_split(X_train,Y_train,test_size=0.3,random_state=2019)



print(x_train.shape)
print(y_train.shape)

print(x_valid.shape)
print(y_valid.shape)

print(x_test.shape)


# In[27]:


# 对标签数据进行one-hot编码

from keras.utils import np_utils
#Y_train = np_utils.to_categorical(Y_train,n_classes)
y_train = np_utils.to_categorical(y_train,n_classes)
y_valid = np_utils.to_categorical(y_valid,n_classes)
print("Shape after one-hot encoding:",y_train.shape)
print("Shape after one-hot encoding:",y_valid.shape)


# ### 模型

# In[28]:


# 导入开发需要的库
from keras import optimizers, Input
from keras.applications import  imagenet_utils

from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.applications import *

from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *


# In[37]:


# 绘制训练过程中的 loss 和 acc 变化曲线
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def history_plot(history_fit):
    plt.figure(figsize=(12,6))
    
    # summarize history for accuracy
    plt.subplot(121)
    plt.plot(history_fit.history["accuracy"])
    plt.plot(history_fit.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper left")
    
    # summarize history for loss
    plt.subplot(122)
    plt.plot(history_fit.history["loss"])
    plt.plot(history_fit.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    
    plt.show()


# In[30]:


# fine-tune 模型
def fine_tune_model(model, optimizer, batch_size, epochs, freeze_num):
    '''
    discription: 对指定预训练模型进行fine-tune，并保存为.hdf5格式
    
    MODEL：传入的模型，VGG16， ResNet50, ...

    optimizer: fine-tune all layers 的优化器, first part默认用adadelta
    batch_size: 每一批的尺寸，建议32/64/128
    epochs: fine-tune all layers的代数
    freeze_num: first part冻结卷积层的数量
    '''

    # datagen = ImageDataGenerator(
    #     rescale=1.255,
    #     # shear_range=0.2,
    #     # zoom_range=0.2,
    #     # horizontal_flip=True,
    #     # vertical_flip=True,
    #     # fill_mode="nearest"
    #   )
    
    # datagen.fit(X_train)
    
    
    # first: 仅训练全连接层（权重随机初始化的）
    # 冻结所有卷积层
    
    for layer in model.layers[:freeze_num]:
        layer.trainable = False
    
    model.compile(optimizer=optimizer, 
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),
    #                     steps_per_epoch=len(x_train)/32,
    #                     epochs=3,
    #                     shuffle=True,
    #                     verbose=1,
    #                     datagen.flow(x_valid, y_valid))
    model.fit(x_train,
         y_train,
         batch_size=batch_size,
         epochs=3,
         shuffle=True,
         verbose=1,
         validation_data=(x_valid,y_valid)
        )
    print('Finish step_1')
    
    
    # second: fine-tune all layers
    for layer in model.layers[freeze_num:]:
        layer.trainable = True
    
    rc = ReduceLROnPlateau(monitor="val_loss",
                factor=0.2,
                patience=3,
                verbose=1,
                mode='min')

    model_name = model.name  + ".hdf5"
    mc = ModelCheckpoint(model_name, 
               monitor="val_loss", 
               save_best_only=True,
               verbose=1,
               mode='min')
    el = EarlyStopping(monitor="val_loss",
              min_delta=0,
              patience=5,
              verbose=1,
              restore_best_weights=True)
    
    model.compile(optimizer=optimizer, 
           loss='categorical_crossentropy', 
           metrics=["accuracy"])

    # history_fit = model.fit_generator(datagen.flow(x_train,y_train,batch_size=32),
    #                                  steps_per_epoch=len(x_train)/32,
    #                                  epochs=epochs,
    #                                  shuffle=True,
    #                                  verbose=1,
    #                                  callbacks=[mc,rc,el],
    #                                  datagen.flow(x_valid, y_valid))
    history_fit = model.fit(x_train,
                 y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 shuffle=True,
                 verbose=1,
                 validation_data=(x_valid,y_valid),
                 callbacks=[mc,rc,el])
    
    print('Finish fine-tune')
    return history_fit


# In[31]:


# 定义一个VGG16的模型
def vgg16_model(img_rows,img_cols):
  x = Input(shape=(img_rows, img_cols, 3))
  x = Lambda(imagenet_utils.preprocess_input)(x)
  base_model = VGG16(input_tensor=x,weights="imagenet",include_top=False, pooling='avg')
  x = base_model.output
  x = Dense(1024,activation="relu",name="fc1")(x)
  x = Dropout(0.5)(x)
  predictions = Dense(n_classes,activation="softmax",name="predictions")(x)

  vgg16_model = Model(inputs=base_model.input,outputs=predictions,name="vgg16")
  
  return vgg16_model


# In[32]:


# 创建VGG16模型
img_rows, img_cols = 224, 224
vgg16_model = vgg16_model(img_rows,img_cols)


# In[33]:


for i,layer in enumerate(vgg16_model.layers):
  print(i,layer.name)


# In[34]:


vgg16_model.summary()


# In[35]:


optimizer = optimizers.Adam(lr=0.0001)
batch_size = 32
epochs = 30
freeze_num = 21


get_ipython().run_line_magic('time', 'vgg16_history = fine_tune_model(vgg16_model,optimizer,batch_size,epochs,freeze_num)')


# In[38]:


history_plot(vgg16_history)


# In[39]:


get_ipython().system('pip install -U efficientnet')


# In[46]:


# 导入Efficient模块
from efficientnet.keras import EfficientNetB4
import keras.backend as K


# In[47]:


# 定义一个EfficientNet模型
def efficient_model(img_rows,img_cols):
  K.clear_session()
  x = Input(shape=(img_rows,img_cols,3))
  x = Lambda(imagenet_utils.preprocess_input)(x)
  
  base_model = EfficientNetB4(input_tensor=x,weights="imagenet",include_top=False,pooling="avg")
  x = base_model.output
  x = Dense(1024,activation="relu",name="fc1")(x)
  x = Dropout(0.5)(x)
  predictions = Dense(n_classes,activation="softmax",name="predictions")(x)

  eB_model = Model(inputs=base_model.input,outputs=predictions,name="eB4")

  return eB_model


# In[48]:


# 创建Efficient模型
img_rows,img_cols=224,224
eB_model = efficient_model(img_rows,img_cols)


# In[49]:


for i,layer in enumerate(eB_model.layers):
  print(i,layer.name)


# In[50]:


eB_model.summary()


# In[ ]:


optimizer = optimizers.Adam(lr=0.0001)
batch_size = 32
epochs = 30
freeze_num = 469
eB_model_history  = fine_tune_model(eB_model,optimizer,batch_size,epochs,freeze_num)


# In[ ]:


history_plot(eB_model_history)


# In[ ]:





# In[ ]:


get_ipython().system('pip install -U efficientnet')


# In[ ]:


# 导入模块
from efficientnet.keras import EfficientNetB3
import keras.backend as K


# In[ ]:


# 定义一个加入Attention模块的Efficient网络架构即efficientnet-with-attention

def efficient_attention_model(img_rows,img_cols):
  K.clear_session()
  
  in_lay = Input(shape=(img_rows,img_cols,3))
  base_model = EfficientNetB3(input_shape=(img_rows,img_cols,3),weights="imagenet",include_top=False)

  pt_depth = base_model.get_output_shape_at(0)[-1]

  pt_features = base_model(in_lay)
  bn_features = BatchNormalization()(pt_features)

  # here we do an attention mechanism to turn pixels in the GAP on an off
  atten_layer = Conv2D(64,kernel_size=(1,1),padding="same",activation="relu")(Dropout(0.5)(bn_features))
  atten_layer = Conv2D(16,kernel_size=(1,1),padding="same",activation="relu")(atten_layer)
  atten_layer = Conv2D(8,kernel_size=(1,1),padding="same",activation="relu")(atten_layer)
  atten_layer = Conv2D(1,kernel_size=(1,1),padding="valid",activation="sigmoid")(atten_layer)# H,W,1
  # fan it out to all of the channels
  up_c2_w = np.ones((1,1,1,pt_depth)) #1,1,C
  up_c2 = Conv2D(pt_depth,kernel_size=(1,1),padding="same",activation="linear",use_bias=False,weights=[up_c2_w])
  up_c2.trainable = False
  atten_layer = up_c2(atten_layer)# H,W,C

  mask_features = multiply([atten_layer,bn_features])# H,W,C

  gap_features = GlobalAveragePooling2D()(mask_features)# 1,1,C
  # gap_mask = GlobalAveragePooling2D()(atten_layer)# 1,1,C

  # # to account for missing values from the attention model
  # gap = Lambda(lambda x:x[0]/x[1],name="RescaleGAP")([gap_features,gap_mask])
  gap_dr = Dropout(0.25)(gap_features)
  dr_steps = Dropout(0.25)(Dense(1000,activation="relu")(gap_dr))
  out_layer = Dense(n_classes,activation="softmax")(dr_steps)
  eb_atten_model = Model(inputs=[in_lay],outputs=[out_layer])

  return eb_atten_model


# In[ ]:


img_rows,img_cols = 224,224
eB_atten_model = efficient_attention_model(img_rows,img_cols)


# In[ ]:


for i,layer in enumerate(eB_atten_model.layers):
  print(i,layer.name)


# In[ ]:


eB_atten_model.summary()


# In[ ]:


optimizer = optimizers.Adam(lr=0.0001)
batch_size = 32
epochs = 30
freeze_num = 12
eB_atten_model_history  = fine_tune_model(eB_atten_model,optimizer,batch_size,epochs,freeze_num)


# In[ ]:


history_plot(eB_atten_model_history)


# In[ ]:





# In[ ]:


get_ipython().system('pip install -U efficientnet')


# In[ ]:


# 导入模块
from efficientnet.keras import EfficientNetB3
import keras.backend as K
import tensorflow as tf


# In[ ]:


from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid

def attach_attention_module(net, attention_module):
  if attention_module == 'se_block': # SE_block
    net = se_block(net)
  elif attention_module == 'cbam_block': # CBAM_block
    net = cbam_block(net)
  else:
    raise Exception("'{}' is not supported attention module!".format(attention_module))

  return net

def se_block(input_feature, ratio=8):
	"""Contains the implementation of Squeeze-and-Excitation(SE) block.
	As described in https://arxiv.org/abs/1709.01507.
	"""
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]

	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel)
	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel//ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel)
	if K.image_data_format() == 'channels_first':
		se_feature = Permute((3, 1, 2))(se_feature)

	se_feature = multiply([input_feature, se_feature])
	return se_feature

def cbam_block(cbam_feature, ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature)
	return cbam_feature

def channel_attention(input_feature, ratio=8):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_feature._keras_shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature._keras_shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool._keras_shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool._keras_shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat._keras_shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
	assert cbam_feature._keras_shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return multiply([input_feature, cbam_feature])


# In[ ]:


# 定义一个EfficientNet模型
def efficient__atten2_model(img_rows,img_cols):
  K.clear_session()
  
  in_lay = Input(shape=(img_rows,img_cols,3))
  base_model = EfficientNetB3(input_shape=(img_rows,img_cols,3),weights="imagenet",include_top=False)
  pt_features = base_model(in_lay)
  bn_features = BatchNormalization()(pt_features)

  atten_features = attach_attention_module(bn_features,"se_block")
  gap_features = GlobalAveragePooling2D()(atten_features)

  gap_dr = Dropout(0.25)(gap_features)
  dr_steps = Dropout(0.25)(Dense(1000,activation="relu")(gap_dr))
  out_layer = Dense(n_classes,activation="softmax")(dr_steps)
  eb_atten_model = Model(inputs=[in_lay],outputs=[out_layer])

  return eb_atten_model


# In[ ]:


img_rows,img_cols = 224,224
eB_atten2_model = efficient__atten2_model(img_rows,img_cols)


# In[ ]:


for i,layer in enumerate(eB_atten2_model.layers):
  print(i,layer.name)


# In[ ]:


eB_atten2_model.summary()


# In[ ]:


optimizer = optimizers.Adam(lr=0.0001)
batch_size = 32
epochs = 30
freeze_num = 19
eB_atten2_model_history  = fine_tune_model(eB_atten2_model,optimizer,batch_size,epochs,freeze_num)


# In[ ]:


history_plot(eB_atten2_model_history)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# 定义双线性VGG16模型

from keras import backend as K

def batch_dot(cnn_ab):
    return K.batch_dot(cnn_ab[0], cnn_ab[1], axes=[1, 1])

def sign_sqrt(x):
    return K.sign(x) * K.sqrt(K.abs(x) + 1e-10)

def l2_norm(x):
    return K.l2_normalize(x, axis=-1)
 
 
def bilinear_vgg16(img_rows,img_cols):
    input_tensor = Input(shape=(img_rows,img_cols,3))
    input_tensor = Lambda(imagenet_utils.preprocess_input)(input_tensor)

    model_vgg16 = VGG16(include_top=False, weights="imagenet",
                        input_tensor=input_tensor,pooling="avg")
    
    cnn_out_a = model_vgg16.layers[-2].output
    cnn_out_shape = model_vgg16.layers[-2].output_shape
    cnn_out_a = Reshape([cnn_out_shape[1]*cnn_out_shape[2],
                         cnn_out_shape[-1]])(cnn_out_a)

    cnn_out_b = cnn_out_a

    cnn_out_dot = Lambda(batch_dot)([cnn_out_a, cnn_out_b])
    cnn_out_dot = Reshape([cnn_out_shape[-1]*cnn_out_shape[-1]])(cnn_out_dot)
 
    sign_sqrt_out = Lambda(sign_sqrt)(cnn_out_dot)
    l2_norm_out = Lambda(l2_norm)(sign_sqrt_out)
    
    fc1 = Dense(1024,activation="relu",name="fc1")(l2_norm_out)
    dropout = Dropout(0.5)(fc1)
    output = Dense(n_classes, activation="softmax",name="output")(dropout)
    bvgg16_model = Model(inputs=model_vgg16.input, outputs=output,name="bvgg16")

    return bvgg16_model


# In[ ]:


# 创建双线性VGG16模型
img_rows,img_cols = 224,224
bvgg16_model = bilinear_vgg16(img_rows,img_cols)


# In[ ]:


for i,layer in enumerate(bvgg16_model.layers):
  print(i,layer.name)


# In[ ]:


optimizer = optimizers.Adam(lr=0.0001)
batch_size = 32
epochs = 100
freeze_num = 25
bvgg16_history = fine_tune_model(bvgg16_model,optimizer,batch_size,epochs,freeze_num)


# In[ ]:


history_plot(bvgg16_history)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### ================================

# In[ ]:




