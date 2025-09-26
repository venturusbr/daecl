import os
import numpy as np
import tensorflow as tf

from skimage.util import random_noise

from tensorflow import keras
from keras.datasets import cifar10, mnist, fashion_mnist
from keras.utils import load_img

#----------------------------------------------------------------------LOAD DATA-------------------------------------------------------------------------------

def loadImages(_FOLDER_DATASET, _DATASET, _PERCENTAGE, _NOISE, noise_level):
	OriginalTrain, OriginalTest = readImages(_FOLDER_DATASET, _DATASET, _PERCENTAGE)
	if _NOISE == "gaussian":
		NoisyTrain, NoisyTest = noiseAdd(OriginalTrain, OriginalTest, _FOLDER_DATASET, _DATASET, noise_level)
	else:
		NoisyTrain, NoisyTest = OriginalTrain, OriginalTest

	print("\t\tOriginal training set: ", OriginalTrain.shape)
	print("\t\tNoisy training set:    ", NoisyTrain.shape)
	print("\t\tOriginal testing set:  ", OriginalTest.shape)
	print("\t\tNoisy testing set:     ", NoisyTest.shape)
	return OriginalTrain, NoisyTrain, OriginalTest, NoisyTest


def readImages(_FOLDER_DATASET, _DATASET, _PERCENTAGE):
	if not os.path.exists('%s%s_OriginalTrain.npy' % (_FOLDER_DATASET, _DATASET)):
		if _DATASET == 'cifar':
			(OriginalTrain, _), (OriginalTest, _) = cifar10.load_data()
		elif _DATASET == 'mnist':
			(OriginalTrain, _), (OriginalTest, _) = mnist.load_data()
			OriginalTrain = np.array(tf.image.grayscale_to_rgb(tf.expand_dims(OriginalTrain, axis=3)))
			OriginalTest = np.array(tf.image.grayscale_to_rgb(tf.expand_dims(OriginalTest, axis=3)))
		elif _DATASET == 'fashion_mnist':
			(OriginalTrain, _), (OriginalTest, _) = fashion_mnist.load_data()
			OriginalTrain = np.array(tf.image.grayscale_to_rgb(tf.expand_dims(OriginalTrain, axis=3)))
			OriginalTest = np.array(tf.image.grayscale_to_rgb(tf.expand_dims(OriginalTest, axis=3)))
		elif _DATASET == 'bsds300':
			OriginalTrain = []
			for root, dirs, files in os.walk('images/BSDS300/train'):
				for f in files:
					img = np.array(load_img(os.path.join(root,f)))
					if (img.shape[0] == 481):
						OriginalTrain.append(np.transpose(img, (1, 0, 2)))
					else:
						OriginalTrain.append(img)
			OriginalTrain = np.array(tf.image.resize(OriginalTrain, [320,480]))

			OriginalTest = []
			for root, dirs, files in os.walk('images/BSDS300/test'):
				for f in files:
					img = np.array(load_img(os.path.join(root,f)))
					if (img.shape[0] == 481):
						OriginalTest.append(np.transpose(img, (1, 0, 2)))
					else:
						OriginalTest.append(img)
			OriginalTest = np.array(tf.image.resize(OriginalTest, [320,480]))
		elif _DATASET == 'fruits360':
			OriginalTrain = []
			for root, dirs, files in os.walk('images/fruits360/Train'):
				for f in files:
					OriginalTrain.append(np.array(load_img(os.path.join(root,f))))
			OriginalTrain = np.array(OriginalTrain)

			OriginalTest = []
			for root, dirs, files in os.walk('images/fruits360/Test'):
				for f in files:
					OriginalTest.append(np.array(load_img(os.path.join(root,f))))
			OriginalTest = np.array(OriginalTest)

		perm = np.random.permutation(OriginalTrain.shape[0])
		OriginalTrain = OriginalTrain[perm[0: (int)(OriginalTrain.shape[0] * (_PERCENTAGE/100))]]
		perm = np.random.permutation(OriginalTest.shape[0])
		OriginalTest = OriginalTest[perm[0: (int)(OriginalTest.shape[0] * (_PERCENTAGE/100))]]

		OriginalTrain = OriginalTrain / 255.0
		OriginalTest = OriginalTest / 255.0

		np.save('%s%s_OriginalTrain.npy' % (_FOLDER_DATASET, _DATASET), OriginalTrain)
		np.save('%s%s_OriginalTest.npy' % (_FOLDER_DATASET, _DATASET), OriginalTest)

	else:
		OriginalTrain = np.load('%s%s_OriginalTrain.npy' % (_FOLDER_DATASET, _DATASET))
		OriginalTest = np.load('%s%s_OriginalTest.npy' % (_FOLDER_DATASET, _DATASET))

	return OriginalTrain, OriginalTest


def noiseAdd(OriginalTrain, OriginalTest, _FOLDER_DATASET, _DATASET, noise_level):
	if not os.path.exists('%s%s_NoisyTrain_%s.npy' % (_FOLDER_DATASET, _DATASET, str(noise_level))):
		NoisyTrain = random_noise(OriginalTrain, mode='gaussian', clip=True, var=noise_level)
		NoisyTest  = random_noise(OriginalTest , mode='gaussian', clip=True, var=noise_level)

		np.save('%s%s_NoisyTrain_%s.npy' % (_FOLDER_DATASET, _DATASET, str(noise_level)), NoisyTrain)
		np.save('%s%s_NoisyTest_%s.npy' % (_FOLDER_DATASET, _DATASET, str(noise_level)), NoisyTest)

	else:
		NoisyTrain = np.load('%s%s_NoisyTrain_%s.npy' % (_FOLDER_DATASET, _DATASET, str(noise_level)))
		NoisyTest = np.load('%s%s_NoisyTest_%s.npy' % (_FOLDER_DATASET, _DATASET, str(noise_level)))

	return NoisyTrain, NoisyTest

#----------------------------------------------------------------------LOAD TEST DATA--------------------------------------------------------------------------

def loadImagesOnlyTest(_FOLDER_DATASET, _DATASET, _PERCENTAGE, _NOISE, noise_level):
	# loading images
	if not os.path.exists('%s%s_OriginalTest.npy' % (_FOLDER_DATASET, _DATASET)):
		if _DATASET == 'kodak24':
			OriginalTest = []
			for root, dirs, files in os.walk('images/Kodak24'):
				for f in files:
					img = np.array(load_img(os.path.join(root,f)))
					if (img.shape[0] > img.shape[1]):
						OriginalTest.append(np.transpose(img, (1, 0, 2)))
					else:
						OriginalTest.append(img)
			OriginalTest = np.array(tf.image.resize(OriginalTest, [320,480]))

		elif _DATASET == 'sunhays80':
			OriginalTest = []
			for root, dirs, files in os.walk('images/SunHays80'):
				for f in files:
					img = np.array(load_img(os.path.join(root,f)))
					if (img.shape[0] > img.shape[1]):
						img = np.transpose(img, (1, 0, 2))
					OriginalTest.append(tf.image.resize(img, [320,480]))
			OriginalTest = np.array(OriginalTest)

		elif _DATASET == 'urban100':
			OriginalTest = []
			for root, dirs, files in os.walk('images/Urban100'):
				for f in files:
					img = np.array(load_img(os.path.join(root,f)))
					if (img.shape[0] > img.shape[1]):
						img = np.transpose(img, (1, 0, 2))
					OriginalTest.append(tf.image.resize(img, [320,480]))
			OriginalTest = np.array(OriginalTest)

		perm = np.random.permutation(OriginalTest.shape[0])
		OriginalTest = OriginalTest[perm[0: (int)(OriginalTest.shape[0] * (_PERCENTAGE/100))]]
		OriginalTest = OriginalTest / 255.0
		np.save('%s%s_OriginalTest.npy' % (_FOLDER_DATASET, _DATASET), OriginalTest)

	else:
		OriginalTest = np.load('%s%s_OriginalTest.npy' % (_FOLDER_DATASET, _DATASET))

	# noising
	if _NOISE == "gaussian":
		if not os.path.exists('%s%s_NoisyTest_%s.npy' % (_FOLDER_DATASET, _DATASET, str(noise_level))):
			NoisyTest  = random_noise(OriginalTest , mode='gaussian', clip=True, var=noise_level)
			np.save('%s%s_NoisyTest_%s.npy' % (_FOLDER_DATASET, _DATASET, str(noise_level)), NoisyTest)
		else:
			NoisyTest = np.load('%s%s_NoisyTest_%s.npy' % (_FOLDER_DATASET, _DATASET, str(noise_level)))
	else:
		NoisyTest = OriginalTest

	print("\t\tOriginal testing set: ", OriginalTest.shape)
	print("\t\tNoisy testing set:    ", NoisyTest.shape)
	return OriginalTest, NoisyTest