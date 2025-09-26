import tensorflow as tf

from tensorflow import keras
from keras import backend as K
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation, AveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam

#------------------------------------------------------------------------BASE MODEL -------------------------------------------------------------------------------

def setBaseModel(OriginalTrain, _NOISE, noise_level):
	print("\t\tLoading the architecture ...")
	tf.random.set_seed(42)

	input_img = Input(shape=(OriginalTrain.shape[1], OriginalTrain.shape[2], OriginalTrain.shape[3]))
	if _NOISE == "dropout_based":
		input_img = Dropout(noise_level)(input_img)	

	encoder = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_img)
	if (OriginalTrain.shape[1] == 28):
		encoder = Conv2D(8, kernel_size=(3, 3), padding='valid', activation='relu')(encoder)
	else:
		encoder = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(encoder) 
	encoder = MaxPooling2D(pool_size=(2, 2))(encoder)
	encoder = Flatten(name='code')(encoder)
	
	if (OriginalTrain.shape[1] == 32):
		decoder = Reshape((4, 4, 8))(encoder)
		decoder = UpSampling2D((2, 2))(decoder)
		decoder = Conv2DTranspose(8, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(decoder) 
	elif (OriginalTrain.shape[1] == 28):
		decoder = Reshape((6, 6, 8))(encoder)
		decoder = UpSampling2D((2, 2))(decoder)
		decoder = Conv2DTranspose(8, kernel_size=(3, 3), padding='valid', activation='relu')(decoder)
	elif (OriginalTrain.shape[1] == 320):
		decoder = Reshape((40, 60, 8))(encoder)
		decoder = UpSampling2D((2, 2))(decoder)
		decoder = Conv2DTranspose(8, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(decoder)

	decoder = Conv2DTranspose(3, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(decoder)
	
	autoencoder = Model(input_img, decoder)
	autoencoder.compile(loss='mean_squared_error', optimizer=Adam())
	#print(autoencoder.summary())

	return autoencoder

#------------------------------------------------------------------------RESIDUAL MODEL --------------------------------------------------------------------------

def setResidualModel(OriginalTrain, _NOISE, noise_level):
	print("\t\tLoading the architecture ...")
	tf.random.set_seed(42)
	
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1

	input_img = Input(shape=(OriginalTrain.shape[1], OriginalTrain.shape[2], OriginalTrain.shape[3]))
	if _NOISE == "dropout_based":
		input_img = Dropout(noise_level)(input_img)	
		
	encoder = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(input_img)
	encoder = BatchNormalization(axis=bn_axis, name='bn_conv1')(encoder)
	encoder = Activation('relu')(encoder)
	encoder = MaxPooling2D((2, 2), strides=(2, 2))(encoder)
	encoder = identity_block(encoder, 3, [64, 64, 64], bn_axis, stage=1, block='a')
	encoder = identity_block(encoder, 3, [64, 64, 64], bn_axis, stage=2, block='b')
	encoder = identity_block(encoder, 3, [64, 64, 64], bn_axis, stage=3, block='c')
	encoder = identity_block(encoder, 3, [64, 64, 64], bn_axis, stage=4, block='d')
	
	if (OriginalTrain.shape[1] == 32):
		encoder = AveragePooling2D((4, 4), name='avg_pool')(encoder) # 8, 8
		encoder = Flatten(name='code')(encoder)
		decoder = Reshape((2, 2, 64))(encoder)# 1, 1, 64
		decoder = UpSampling2D((4, 4))(decoder) # 8,8
	elif (OriginalTrain.shape[1] == 28):
		encoder = AveragePooling2D((7, 7), name='avg_pool')(encoder)
		encoder = Flatten(name='code')(encoder)
		decoder = Reshape((1, 1, 64))(encoder)
		decoder = UpSampling2D((7, 7))(decoder) 
	elif (OriginalTrain.shape[1] == 320):
		encoder = AveragePooling2D((10, 10), name='avg_pool')(encoder)
		encoder = Flatten(name='code')(encoder)
		decoder = Reshape((8, 12, 64))(encoder)
		decoder = UpSampling2D((10, 10))(decoder)
	elif (OriginalTrain.shape[1] == 100):
		encoder = AveragePooling2D((5, 5), name='avg_pool')(encoder)
		encoder = Flatten(name='code')(encoder)
		decoder = Reshape((5, 5, 64))(encoder)
		decoder = UpSampling2D((5, 5))(decoder)
	
	decoder = identity_blockDecoder(decoder, 3, [64, 64, 64], bn_axis, stage=4, block='d_dec')
	decoder = identity_blockDecoder(decoder, 3, [64, 64, 64], bn_axis, stage=3, block='c_dec')
	decoder = identity_blockDecoder(decoder, 3, [64, 64, 64], bn_axis, stage=2, block='b_dec')
	decoder = identity_blockDecoder(decoder, 3, [64, 64, 64], bn_axis, stage=1, block='a_dec')
	decoder = UpSampling2D((2, 2))(decoder)
	decoder = BatchNormalization()(decoder)
	decoder = Conv2DTranspose(3, kernel_size=(7,7), strides=(2, 2), padding='same', activation='relu')(decoder)

	autoencoder = Model(input_img, decoder)
	autoencoder.compile(loss='mean_squared_error', optimizer=Adam())
	#print(autoencoder.summary())
	
	return autoencoder


def identity_block(input_tensor, kernel_size, filters, bn_axis, stage, block):
	filters1, filters2, filters3 = filters

	x = Conv2D(filters1, (1, 1), name='res'+str(stage)+block+'_branch2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name='bn'+str(stage)+block+'_branch2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size, padding='same', name='res'+str(stage)+block+'_branch2b')(x)
	x = BatchNormalization(axis=bn_axis, name='bn'+str(stage)+block+'_branch2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1, 1), name='res'+str(stage)+block+'_branch2c')(x)
	x = BatchNormalization(axis=bn_axis, name='bn'+str(stage)+block+'_branch2c')(x)
	x = layers.add([x, input_tensor])
	x = Activation('relu')(x)
	
	return x


def identity_blockDecoder(input_tensor, kernel_size, filters, bn_axis, stage, block):
	filters1, filters2, filters3 = filters

	x = BatchNormalization(axis=bn_axis, name='bn'+str(stage)+block+'_branch2c')(input_tensor)
	x = Conv2DTranspose(filters3, (1, 1), name='res'+str(stage)+block+'_branch2c')(x)
	x = Activation('relu')(x)

	x = BatchNormalization(axis=bn_axis, name='bn'+str(stage)+block+'_branch2b')(x)
	x = Conv2DTranspose(filters2, kernel_size, padding='same', name='res'+str(stage)+block+'_branch2b')(x)
	x = Activation('relu')(x)

	x = BatchNormalization(axis=bn_axis, name='bn'+str(stage)+block+'_branch2a')(x)
	x = Conv2DTranspose(filters1, (1, 1), name='res'+str(stage)+block+'_branch2a')(x)
	x = Activation('relu')(x)
	
	return x