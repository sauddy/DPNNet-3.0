# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import layers

## Modules for hyperparameter tuning keras-tuner
import keras_tuner as kt
from keras_tuner import HyperModel
import tensorflow as tf


# modules added for the RESNET50
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import ZeroPadding2D, AveragePooling2D, GlobalMaxPooling2D, Add


######## Alexnet


def alexnet(width, height, depth, classes=None,regress=False,multi_label=False,classification=False,option=None):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)


# define the model input
    X_input = Input(shape=inputShape)
    # X_input = Input(shape=inputShape)

# 1st Convolutional Layer
    x = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', padding="same")(X_input)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)

# 2nd Convolutional Layer
    x = Conv2D(256, kernel_size=(11, 11), strides=(1, 1), activation='relu', padding="same")(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)

# 3rd Convolutional Layer
    x = Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)

# 4th Convolutional Layer
    x = Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)

# 5th Convolutional Layer
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)


# Passing it to a Fully Connected layer
    x = Flatten()(x)
# 1st Fully Connected Layer
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
# Add Dropout to prevent overfitting
    x = Dropout(0.4)(x)

# 2nd Fully Connected Layer
    x = Dense(4096, activation='relu')(x)
# Add Dropout
    x = Dropout(0.4)(x)

# 3rd Fully Connected Layer
    x = Dense(1000, activation='relu')(x)
    x = BatchNormalization(axis=1)(x)
# Add Dropout
    x = Dropout(0.4)(x)

# # Output Layer
#     x = Dense(4)(x)
#     # x=Activation("relu")(x)

    XX = Flatten()(x)
   
    if classification==True:
      # print("A new fully Connexted layer is added")
       ## Adding a Fully connected layer for classification
      Xclas = XX
      
      Xclas = Dense(100, kernel_initializer='he_uniform', activation='relu')(Xclas)
      # Xclas = BatchNormalization(axis=1)(Xclas) 
      print("INFO:CNN is used for classification")
      if option ==1:
        print("Additional INFO:Using Softmax for classification")
        Xclas = Dense(units=classes, activation='softmax')(Xclas) 
      else:  
        print("Additional INFO: Using Sigmoid activation for classification")
        Xclas = Dense(units=classes, name='cla',activation='sigmoid')(Xclas)
      out_clas = Xclas 

    if regress == True:
      print("INFO:CNN is used for regression")
      if multi_label==False:           
        Xreg = Dense(1, activation='linear', name='reg', kernel_initializer = glorot_uniform(seed=0))(XX)
      if multi_label==True: 
        ## 28 Feb 2022 added the multi-label output 
        print("INFO: Note Multiple Labels are optimised during regression")
        Xreg = Dense(3, activation='linear', name='reg', kernel_initializer = glorot_uniform(seed=0))(XX)
        out_reg = Xreg
    # else:
    #     print("Multi-input is initiated")
    #     X = Dense(4)(X)
    #     X = Activation("relu")(X)
    

    # Create model

    if classification == True and regress == True:
      print("INFO: Performing both regression and classification -- Model Training")
      modelalexnet = Model(inputs = X_input,outputs=[out_reg, out_clas])
    elif regress == False:
      X = out_clas
      print("INFO: Classification Model is being trained")
      modelalexnet = Model(inputs = X_input, outputs = X)
    elif classification is False:
      X = out_reg
      print("INFO: Regression Model is being trained")
      modelalexnet = Model(inputs = X_input, outputs = X)

    return modelalexnet




############ vgg 16


def cnn_vgg(width, height, depth, classes=None,regress=False,multi_label=False,classification=False,option=None):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)


    # define the model input
    X_input = Input(shape=inputShape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(X_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = BatchNormalization(axis=1)(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    
    
    XX = Flatten()(x)
   
    if classification==True:
      # print("A new fully Connexted layer is added")
       ## Adding a Fully connected layer for classification
      Xclas = XX
      
      Xclas = Dense(100, kernel_initializer='he_uniform', activation='relu')(Xclas)
      # Xclas = BatchNormalization(axis=1)(Xclas) 
      print("INFO:CNN is used for classification")
      if option ==1:
        print("Additional INFO:Using Softmax for classification")
        Xclas = Dense(units=classes, activation='softmax')(Xclas) 
      else:  
        print("Additional INFO: Using Sigmoid activation for classification")
        Xclas = Dense(units=classes, name='cla',activation='sigmoid')(Xclas)
      out_clas = Xclas 

    if regress == True:
      print("INFO:CNN is used for regression")
      if multi_label==False:           
        Xreg = Dense(1, activation='linear', name='reg', kernel_initializer = glorot_uniform(seed=0))(XX)
      if multi_label==True: 
        ## 28 Feb 2022 added the multi-label output 
        print("INFO: Note Multiple Labels are optimised during regression")
        Xreg = Dense(3, activation='linear', name='reg', kernel_initializer = glorot_uniform(seed=0))(XX)
        out_reg = Xreg
    # else:
    #     print("Multi-input is initiated")
    #     X = Dense(4)(X)
    #     X = Activation("relu")(X)
    

    # Create model

    if classification == True and regress == True:
      print("INFO: Performing both regression and classification -- Model Training")
      modelvgg = Model(inputs = X_input,outputs=[out_reg, out_clas])
    elif regress == False:
      X = out_clas
      print("INFO: Classification Model is being trained")
      modelvgg = Model(inputs = X_input, outputs = X)
    elif classification is False:
      X = out_reg
      print("INFO: Regression Model is being trained")
      modelvgg = Model(inputs = X_input, outputs = X)

    return modelvgg

####### RESNET50

#identity_block

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X


def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a',padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b', padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c', padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X
def Resnet50(width, height, depth,classes=None,regress=False,multi_label=False,classification=False,option=None):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    inputShape = (height, width, depth)
    

    # define the model input
    X_input = Input(shape=inputShape)



    # Define the input as a tensor with shape input_shape
    #X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [32, 32, 128], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [32, 32, 128], stage=2, block='b')
    X = identity_block(X, 3, [32, 32, 128], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [64, 64, 256], stage=3, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=3, block='c')
    X = identity_block(X, 3, [64, 64, 256], stage=3, block='d')

    # Stage 4 
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='d')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='e')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='f')

    # Stage 5 
    X = convolutional_block(X, f = 3, filters = [256,256, 1024], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [256,256, 1024], stage=5, block='b')
    X = identity_block(X, 3, [256,256, 1024], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2), name='avg_pool')(X)
    
    ## made the changes on 15 Feb 2021
    # output layer
    XX = Flatten()(X)
    # XX = layers.GlobalAveragePooling2D(name="avg_pool")(X)
    XX = Dense(100, kernel_initializer='he_uniform', activation='relu')(XX) 
   
    if classification==True:
      # print("A new fully Connexted layer is added")
       ## Adding a Fully connected layer for classification
      Xclas = XX
      Xclas = BatchNormalization(axis=1)(Xclas)   
      top_dropout_rate = 0.30
      # Xclas = layers.Dropout(top_dropout_rate, name="top_dropout1")(Xclas)  
      Xclas = Dense(50, kernel_initializer='he_uniform', activation='relu')(Xclas)
      Xclas = BatchNormalization(axis=1)(Xclas)
      # top_dropout_rate = 0.20
      # Xclas = layers.Dropout(top_dropout_rate, name="top_dropout2")(Xclas)
      Xclas = Dense(20, kernel_initializer='he_uniform', activation='relu')(Xclas)
      Xclas = BatchNormalization(axis=1)(Xclas)
        
      print("INFO:CNN is used for classification")
      if option ==1:
        print("Additional INFO:Using Softmax for classification")
        Xclas = Dense(units=classes, activation='softmax')(Xclas) 
      else:  
        print("Additional INFO: Using Sigmoid activation for classification")
        Xclas = Dense(units=classes, name='cla',activation='sigmoid')(Xclas)
      out_clas = Xclas 

    if regress == True:
      print("INFO:CNN is used for regression")
      if multi_label==False:           
        Xreg = Dense(1, activation='linear', name='reg', kernel_initializer = glorot_uniform(seed=0))(XX)
      if multi_label==True: 
        ## 28 Feb 2022 added the multi-label output 
        print("INFO: Note Multiple Labels are optimised during regression")
        Xreg = Dense(3, activation='linear', name='reg', kernel_initializer = glorot_uniform(seed=0))(XX)
        out_reg = Xreg
    # else:
    #     print("Multi-input is initiated")
    #     X = Dense(4)(X)
    #     X = Activation("relu")(X)
    
   

    #### Create model ###

    if classification == True and regress == True:
      print("INFO: Performing both regression and classification -- Model Training")
      modelresnet = Model(inputs = X_input,outputs=[out_reg, out_clas])
    elif regress == False:
      X = out_clas
      print("INFO: Classification Model is being trained")
      modelresnet = Model(inputs = X_input, outputs = X)
    elif classification is False:
      X = out_reg
      print("INFO: Regression Model is being trained")
      modelresnet = Model(inputs = X_input, outputs = X)

    return modelresnet




class TRANSFERLEARNINGHYPERMODEL(HyperModel):
    
    def __init__(self,width,height,depth,classes=None,regress=False,multi_label=False,classification=False,option=None,transfer_model= None,init_lr=0.001,epochs=10):
          self.width = width
          self.height = height
          self.depth = depth
          self.classes = classes
          self.regress = regress
          self.multi_label = multi_label
          self.classification = classification
          self.option = option
          self.transfer_model = transfer_model
          self.init_lr = init_lr
          self.epochs = epochs
          print(self.transfer_model)
    
    def build(self, hp):

          '''
          This class is used to tune the hyperparameters for the classification part. It finds the hyperparemter for the layers succedign 
          Transfer learning network. train the DPNNet Model
          Here we use the pretrained weights from the Imagenet data set but are also further trained
          
          
          Please note this here update the final layers to perform regress or classification
          or both regression and classification together
          
          Input:  1. Dimension of the image [width,height,depth]
                  2. Number of classsification classes if used for calssification
                  3. Regress on or off
                  4. Multilabel -- True for multi-label regression Default == False
                  5. Classification -- True or False 
                  6. Option is the argument to use either softmax (default) or sigmoid for classification
                  7. Select the transfer Learning Model from Either 3 RESNET50 or Efficientnet0-7 models
                  8. intial lr , default is 0.001
                  9. the number of epochs for training. default in 10
                  
                  
          Output : Model Build 
          
          '''
          
          ## Input layers
          X_input = layers.Input(shape=(self.height, self.width, self.depth))
          x = X_input
          
          if self.transfer_model == None:
              self.transfer_model = EfficientNetB6
              print("Please Select the Network that will be used-- By Default {} will be used".format(str.lower(self.transfer_model.__name__)))
          
          #### Selecting the model along with the pretrained weights
          mod_name = self.transfer_model.__name__
          print("INFO: Keras Model used is {}".format(mod_name))
          print("Loading weights for {} since this notebook is not connected to internet".format(str.lower(mod_name)))
          
          try:
            model = self.transfer_model(include_top=False,input_tensor=x, weights='preloaded_weights/'+str.lower(mod_name)+'_notop.h5')
          except ValueError:
            model =self.transfer_model(include_top=False,input_tensor=x,weights='preloaded_weights/'+str.lower(mod_name)+'_weights_tf_dim_ordering_tf_kernels_notop.h5')
      
              
          print("IF the notebook is connected to internet then the weights can be directly downloaded-- Choose accordingly")
    
#           try:
#               model = self.transfer_model(include_top=False,input_tensor=x, weights='imagenet')
#           except ValueError:
#               model = self.transfer_model(include_top=False,input_tensor=x, weights='imagenet')
        
          
          # Freeze the pretrained weights
          model.trainable = True

          # Rebuild top

          x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
          x = layers.BatchNormalization()(x)

          top_dropout_rate = 0.2
          x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)

          XX = Flatten()(x)
          XX = Dense(100, kernel_initializer='he_uniform', activation='relu')(XX) 
          # outputs = layers.Dense(classes, activation="softmax", name="pred")(x)


          if self.classification==True:
            print("INFO:CNN is used for regression ")
            # print("A new fully Connexted layer is added")
            ## Adding a Fully connected layer for classification
            Xclas = XX
           

            for i in range(hp.Int('num_of_layers',min_value=1,max_value=8,step=1)):         
              #providing range for number of neurons in hidden layers
              Xclas = Dense(units=hp.Int('num_of_neurons_'+ str(i),min_value=10,max_value=40,step=5), kernel_initializer='he_uniform', activation='relu')(Xclas)
          

            Xclas = BatchNormalization(axis=1)(Xclas)

            Xclas = Dropout(hp.Float('dropout_rate',min_value=0, max_value=0.4, step=0.05), name="classification_dropout")(Xclas)

            Xclas = Dense(units=hp.Int('num_of_neurons',min_value=10,max_value=40,step=5), activation='relu')(Xclas)

            if hp.Choice('activation', ['softmax', 'sigmoid']) == 'softmax':
              Xclas = Dense(units=self.classes, activation='softmax', name='clas')(Xclas)
            else:
              Xclas = Dense(units=self.classes, activation='sigmoid', name='clas')(Xclas)

            out_clas = Xclas 


          if self.regress == True:
            print("INFO:CNN is used for regression")
            if self.multi_label==False:           
              Xreg = Dense(1, activation='linear', name='reg', kernel_initializer = glorot_uniform(seed=0))(XX)
            if self.multi_label==True: 
              ## 28 Feb 2022 added the multi-label output 
              print("INFO: Note Multiple Labels are optimised during regression")
              Xreg = Dense(3, activation='linear', name='reg', kernel_initializer = glorot_uniform(seed=0))(XX)
            
            out_reg = Xreg
          

          ### Create model

          optimizer = tf.keras.optimizers.Adam(self.init_lr, decay=self.init_lr/self.epochs)

          if self.classification == True and self.regress == True:
            print("INFO: Performing both regression and classification -- Model Training")
            modelNEWCNN = Model(inputs = X_input,outputs=[out_reg, out_clas])
            modelNEWCNN.compile(loss=['mean_squared_error','binary_crossentropy'],optimizer=optimizer,metrics=['mean_squared_error', 'accuracy'])
          elif self.regress == False:
            X = out_clas
            print("INFO: Classification Model is being trained")
            modelNEWCNN = Model(inputs = X_input, outputs = X)
            modelNEWCNN.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
          elif self.classification is False:
            X = out_reg
            print("INFO: Regression Model is being trained")
            modelNEWCNN = Model(inputs = X_input, outputs = X)
            modelNEWCNN.compile(loss='mean_squared_error',
                    optimizer=optimizer,
                    metrics=['mean_absolute_error', 'mean_squared_error'])

          return modelNEWCNN
    
    
    
def TRANSFERLEARNING(X_res, Y_res,depth,classes=None,regress=False,multi_label=False,classification=False,option=None,transfer_model= None):

    '''
    This function introduces the use of Transfer learning to train the DPNNet Model
    Here we use the pretrained weights from the Imagenet data set
    
    Please note this here update the final layers to perform regress or classification
    or both regression and classification together
    
    Input:  1. Dimension of the image [width,height,depth]
            2. Number of classsification classes if used for calssification
            3. Regress on or off
            4. Multilabel -- True for multi-label regression Default == False
            5. Classification -- True or False 
            6. Option is the argument to use either softmax (default) or sigmoid for classification
            7. Select the transfer Learning Model from Either 3 RESNET50 or Efficientnet0-7 models
            
            
    Output : Model Build 
    
    '''
    
    ## Input layers
    X_input = layers.Input(shape=(X_res, Y_res, depth))
    x = X_input
    
    if transfer_model == None:
        transfer_model = EfficientNetB6
        print("Please Select the Network that will be used-- By Default {} will be used".format(str.lower(transfer_model.__name__)))
    
    #### Selecting the model along with the pretrained weights
    mod_name = transfer_model.__name__
    print("INFO: Keras Model used is {}".format(mod_name))
    print("Loading weights for {} since this notebook is not connected to internet".format(str.lower(mod_name)))
    
    try:
        model = transfer_model(include_top=False,input_tensor=x, weights='preloaded_weights/'+str.lower(mod_name)+'_notop.h5')
    except ValueError:
        model = transfer_model(include_top=False,input_tensor=x, weights='preloaded_weights/'+str.lower(mod_name)+'_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
    print("IF the notebook is connected to internet then the weights can be directly downloaded-- Choose accordingly")
    # try:
    #     model = transfer_model(include_top=False,input_tensor=x, weights='imagenet')
    # except ValueError:
    #     model = transfer_model(include_top=False,input_tensor=x, weights='imagenet')
    
    # Freeze the pretrained weights
    model.trainable = True

    # Rebuild top

    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)

    XX = Flatten()(x)
    XX = Dense(100, kernel_initializer='he_uniform', activation='relu')(XX) 
    # outputs = layers.Dense(classes, activation="softmax", name="pred")(x)


    if classification==True:
      # print("A new fully Connexted layer is added")
       ## Adding a Fully connected layer for classification
      Xclas = XX
      # Xclas = BatchNormalization(axis=1)(Xclas)   
      top_dropout_rate = 0.30
      # Xclas = layers.Dropout(top_dropout_rate, name="top_dropout1")(Xclas)  
      Xclas = Dense(50, kernel_initializer='he_uniform', activation='relu')(Xclas)
      Xclas = BatchNormalization(axis=1)(Xclas)
      # top_dropout_rate = 0.20
      # Xclas = layers.Dropout(top_dropout_rate, name="top_dropout2")(Xclas)
      Xclas = Dense(20, kernel_initializer='he_uniform', activation='relu')(Xclas)
      Xclas = BatchNormalization(axis=1)(Xclas)
        
      print("INFO:CNN is used for classification")
      if option ==1:
          print("Additional INFO:Using Softmax for classification")
          Xclas = Dense(units=classes,name='cla',activation='softmax')(Xclas) 
      else:  
          print("Additional INFO: Using Sigmoid activation for classification")
          Xclas = Dense(units=classes, name='cla',activation='sigmoid')(Xclas)
          out_clas = Xclas 


    if regress == True:
      print("INFO:CNN is used for regression")
      if multi_label==False:           
        Xreg = Dense(1, activation='linear', name='reg', kernel_initializer = glorot_uniform(seed=0))(XX)
      if multi_label==True: 
        ## 28 Feb 2022 added the multi-label output 
        print("INFO: Note Multiple Labels are optimised during regression")
        Xreg = Dense(3, activation='linear', name='reg', kernel_initializer = glorot_uniform(seed=0))(XX)
        out_reg = Xreg
    

    ### Create model

    if classification == True and regress == True:
      print("INFO: Performing both regression and classification -- Model Training")
      modelNEWCNN = Model(inputs = X_input,outputs=[out_reg, out_clas])
    elif regress == False:
      X = out_clas
      print("INFO: Classification Model is being trained")
      modelNEWCNN = Model(inputs = X_input, outputs = X)
    elif classification is False:
      X = out_reg
      print("INFO: Regression Model is being trained")
      modelNEWCNN = Model(inputs = X_input, outputs = X)

    return modelNEWCNN
