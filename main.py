import os
import time
import math
import cv2
import numpy as np

# AE architecture - training
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import load_model
from skimage.metrics import mean_squared_error as rmse
from keras.optimizers import Adam

# external files
from dataset import loadImages, loadImagesOnlyTest
from dae import setBaseModel, setResidualModel
from measures import mean_measures, checkHardnessScoring, performanceGraphs

#------------------------------------------------------------------------DEFINES-------------------------------------------------------------------------------

# training setup
_NUMBER_EPOCHS = 30
_BATCH_SIZE = 32
_ARCHITECTURE = "residual" # base, residual
_NOISE = "dropout_based" #gaussian, dropout_based

# curriculum learning setup
_NUMBER_STEPS_LADDER = 5
_ANTI_CL = False # False, True
_BOOTSTRAP_FACTOR = int(_NUMBER_EPOCHS * 0.2)
_SCORING_METRIC = ['homoIndex', 'rmse']

# dataset setup
_DATASET = 'cifar' #cifar, mnist, fashion_mnist, bsds300, kodak24, urban100, sunhays80, fruits360
_PERCENTAGE = 100
_NOISE_LEVEL = [0.1, 0.2]
_FOLDER_MODELS = 'models_'+_DATASET+'/'+_ARCHITECTURE+'/'
_FOLDER_DATASET = 'datasets/'
_NUMBER_IMAGES_SAVE = 10
_CROSS_DATASET = 'bsds300' #bsds300

#----------------------------------------------------------------------LOAD DATA-------------------------------------------------------------------------------

def loadData(noise_level, scoreMeasure, folderName):
	print("\tLoading dataset " + _DATASET + " with noise " + str(noise_level) + " ... ")
	OriginalTrain, NoisyTrain, OriginalTest, NoisyTest = loadImages(_FOLDER_DATASET, _DATASET, _PERCENTAGE, _NOISE, noise_level)

	print("\tScoring dataset " + _DATASET + " with noise " + str(noise_level) + " ... ")
	OriginalTrain, NoisyTrain = scoringBootstrap(OriginalTrain, NoisyTrain, noise_level, scoreMeasure, folderName)

	return OriginalTrain, NoisyTrain, OriginalTest, NoisyTest

def loadDataOnlyTest(noise_level, scoreMeasure, folderName):
	print("\tLoading dataset " + _DATASET + " with noise " + str(noise_level) + " ... ")
	OriginalTest, NoisyTest = loadImagesOnlyTest(_FOLDER_DATASET, _DATASET, _PERCENTAGE, _NOISE, noise_level)
	return OriginalTest, NoisyTest

#------------------------------------------------------------------------SCORING-------------------------------------------------------------------------------

def scoring_rmse(original, pred):
	return rmse(original, pred)


def scoring_homoIndex(img):
	homogeneity = []
	for k in range(0, img.shape[2]):
		h = 0
		for i in range(0, img.shape[0]):
			for j in range(0, img.shape[1]):
				h += (img[i][j][k]) / (1 + abs(i-j))
		homogeneity.append(h)

	return max(homogeneity)


def scoringBootstrap(OriginalTrain, NoisyTrain, noise_level, scoreMeasure, folderName):
	if scoreMeasure == 'rmse':
		if not os.path.exists('%sbootstrap.weights.h5' % folderName):
			print("\t\tTraining scoring model ...")
			bootstrapModel = setModel(OriginalTrain, noise_level)
			
			start_bootstrap_training = time.time()
			history = bootstrapModel.fit(x=NoisyTrain, y=OriginalTrain, batch_size=_BATCH_SIZE, epochs=_BOOTSTRAP_FACTOR, shuffle=True, verbose=1)
			finish_bootstrap_training = time.time()
			
			bootstrapModel.save_weights('%sbootstrap.weights.h5' % folderName)
			bootstrapModel.save('%sbootstrap.model.h5' % folderName)

			with open('%sbootstrap.txt' % folderName, "a") as text_file:
				text_file.write("Bootstrap Training Time: %ss \n" % str(finish_bootstrap_training - start_bootstrap_training))
		
		else:
			print("\t\tLoading scoring model ...")
			bootstrapModel = load_model('%sbootstrap.model.h5' % folderName)
			bootstrapModel.load_weights('%sbootstrap.weights.h5' % folderName)

		start_time_prediction = time.time()
		prediction = bootstrapModel.predict(NoisyTrain)
		finish_time_prediction = time.time()
		del bootstrapModel

		with open('%sbootstrap.txt' % folderName, "a") as text_file:
			text_file.write("Bootstrap Prediction Time: %ss \n" % str(finish_time_prediction - start_time_prediction))

	print("\t\tScoring images ...")
	score = []

	start_time_score = time.time()
	for i in range(0, NoisyTrain.shape[0]):
		if scoreMeasure == 'rmse':
			score.append(scoring_rmse(NoisyTrain[i], prediction[i]))
		elif scoreMeasure == 'homoIndex':
			score.append(scoring_homoIndex(NoisyTrain[i]))
		else:
			raise Exception('Inexistent Metric!')
	finish_time_score = time.time()

	print("\t\tSorting images ...")
	NoisyTrain = NoisyTrain.tolist()
	OriginalTrain = OriginalTrain.tolist()

	start_time_sort = time.time()
	NoisyTrainSorted = [x for _, x in sorted(zip(score,NoisyTrain), reverse=_ANTI_CL)]
	OriginalTrainSorted = [y for _, y in sorted(zip(score,OriginalTrain), reverse=_ANTI_CL)]
	finish_time_sort = time.time()
	del OriginalTrain, NoisyTrain

	with open('%sscoring.txt' % folderName, "a") as text_file:
		text_file.write("Scoring Time: %ss \n" % str(finish_time_score - start_time_score))
		text_file.write("Sort Time: %ss \n" % str(finish_time_sort - start_time_sort))
	
	return np.array(OriginalTrainSorted), np.array(NoisyTrainSorted)

#-------------------------------------------------------------------------PACING-------------------------------------------------------------------------------

def pacingFunction(NoisyTrain, OriginalTrain, pacing, currentEpoch):
	if (pacing == "linear"):
		size = int((NoisyTrain.shape[0]/_NUMBER_EPOCHS) * currentEpoch)
	
	elif (pacing == "modified_linear"):
		size = int((NoisyTrain.shape[0]*0.5/_NUMBER_EPOCHS) * currentEpoch + NoisyTrain.shape[0]*0.5)

	elif (pacing == "log"):
		size = int((math.log((currentEpoch/_NUMBER_EPOCHS)*(NoisyTrain.shape[0]),(NoisyTrain.shape[0])))*(NoisyTrain.shape[0]))
	
	elif (pacing == "ladder"):
		sizeStep = int(_NUMBER_EPOCHS/_NUMBER_STEPS_LADDER)
		currentStep = int(currentEpoch/sizeStep)+1
		size = int((NoisyTrain.shape[0]*(currentStep/_NUMBER_STEPS_LADDER)))

	elif (pacing == "modified_ladder"):
		sizeStep = int(_NUMBER_EPOCHS/_NUMBER_STEPS_LADDER)
		currentStep = int(currentEpoch/sizeStep)+1
		size = int((NoisyTrain.shape[0]*0.5*(currentStep/_NUMBER_STEPS_LADDER)) + NoisyTrain.shape[0]*0.5)
	
	elif (pacing == "ladderlog"):
		sizeStep = int(_NUMBER_EPOCHS/_NUMBER_STEPS_LADDER)
		currentStep = (int(currentEpoch/sizeStep)+1)*sizeStep
		size = int((math.log((currentStep/_NUMBER_EPOCHS)*(NoisyTrain.shape[0]),(NoisyTrain.shape[0])))*(NoisyTrain.shape[0]))

	if (size <= 0):
		return NoisyTrain[:1], OriginalTrain[:1]
	else:
		return NoisyTrain[:size], OriginalTrain[:size]

#-------------------------------------------------------------------ARCHITECTURE-------------------------------------------------------------------------------

def setModel(OriginalTrain, noise_level):
	if _ARCHITECTURE == "base":
		return setBaseModel(OriginalTrain, _NOISE, noise_level)
	elif _ARCHITECTURE == "residual":
		return setResidualModel(OriginalTrain, _NOISE, noise_level)


def trainingModel(OriginalTrain, NoisyTrain, pacing, folderName, noise_level):
	print("\tTraining curriculum model with " + pacing + "...")
	model = setModel(OriginalTrain, noise_level)

	errTrain = []
	sizes = []
	errTrain.append(1)
	sizes.append(0)
	backpropagation = 0
	images = 0

	start_time_training = time.time()
	for e in range(0, _NUMBER_EPOCHS):
		if pacing == "constant":
			Xcurriculum, Ycurriculum = NoisyTrain, OriginalTrain
		else:
			Xcurriculum, Ycurriculum = pacingFunction(NoisyTrain, OriginalTrain, pacing, e+1)
			
		loss = model.fit(x=Xcurriculum, y=Ycurriculum, batch_size=_BATCH_SIZE, epochs=1, shuffle=True, verbose=1)
					
		errTrain.append(float(loss.history['loss'][0]))
		sizes.append(Xcurriculum.shape[0])
		images = images + Xcurriculum.shape[0]
		backpropagation = backpropagation + math.ceil(Xcurriculum.shape[0]/_BATCH_SIZE)
		print("\t\tEpoch %i/%i (%i/%i images): Loss (%.5f). Accumulate Training Time (%.5f)." % 
				(e+1, _NUMBER_EPOCHS, Xcurriculum.shape[0], NoisyTrain.shape[0], float(loss.history['loss'][0]), time.time() - start_time_training))
	finish_time_training = time.time()

	model.save_weights('%s%s.weights.h5' % (folderName, pacing))
	model.save('%s%s.model.h5' % (folderName, pacing))

	with open('%s%s_time.txt' % (folderName, pacing), "a") as text_file:
		text_file.write("DAE Training Time: %ss \n" % str(finish_time_training - start_time_training))
		text_file.write("Number of backpropagation: %s \n" % str(backpropagation))
		text_file.write("Number of images: %s \n" % str(images))
	with open('%s%s_loss.txt' % (folderName, pacing), "a") as text_file:
		text_file.write(str(errTrain))
	with open('%s%s_size.txt' % (folderName, pacing), "a") as text_file:
		text_file.write(str(sizes))


def predictingModel(OriginalTest, NoisyTest, pacing, folderName):
	print("\tPredict curriculum model with " + pacing + "...")
	model = load_model('%s%s.model.h5' % (folderName, pacing))
	model.load_weights('%s%s.weights.h5' % (folderName, pacing))
	
	start_time_prediction = time.time()
	prediction = model.predict(NoisyTest)
	finish_time_prediction = time.time()

	folderName = folderName.replace("bsds300", _DATASET)
	with open('%s%s_time.txt' % (folderName, pacing), "a") as text_file:
		text_file.write("DAE Prediction Time: %ss \n" % str(finish_time_prediction - start_time_prediction))
		
	result_rmse, result_psnr, result_ssim = mean_measures(OriginalTest, prediction)
	print("\t\tMean RMSE: " + str(result_rmse))
	print("\t\tMean PSNR: " + str(result_psnr))
	print("\t\tMean SSIM: " + str(result_ssim))
	
	with open('%s%s_measures.txt' % (folderName, pacing), "a") as text_file:
		text_file.write("Mean RMSE: %s \n" % str(result_rmse))
		text_file.write("Mean PSNR: %s \n" % str(result_psnr))
		text_file.write("Mean SSIM: %s \n" % str(result_ssim))

	if not os.path.exists(folderName+"/examples_"+pacing+"/"):
		os.makedirs(folderName+"/examples_"+pacing+"/")
	for i in range(0, _NUMBER_IMAGES_SAVE):
		cv2.imwrite("%s/examples_%s/%s_noisy.jpg" % (folderName, pacing, str(i)), NoisyTest[i]*255)
		cv2.imwrite("%s/examples_%s/%s_original.jpg" % (folderName, pacing, str(i)), OriginalTest[i]*255)
		cv2.imwrite("%s/examples_%s/%s_prediction.jpg" % (folderName, pacing, str(i)), prediction[i]*255)


def trainingHardnessScoring(OriginalTrain, NoisyTrain, noise_level):
	print("\tChecking the hardness score...")
	model1 = setModel(OriginalTrain, noise_level)
	history1 = model1.fit(x=NoisyTrain[:int(NoisyTrain.shape[0]*0.25)], 
  						y=OriginalTrain[:int(OriginalTrain.shape[0]*0.25)], 
  						batch_size=_BATCH_SIZE, epochs=_NUMBER_EPOCHS, shuffle=True, verbose=1)
	del model1
	
	model2 = setModel(OriginalTrain, noise_level)
	history2 = model2.fit(x=NoisyTrain[int(NoisyTrain.shape[0]*0.25):int(NoisyTrain.shape[0]*0.5)], 
  						y=OriginalTrain[int(OriginalTrain.shape[0]*0.25):int(OriginalTrain.shape[0]*0.5)], 
  						batch_size=_BATCH_SIZE, epochs=_NUMBER_EPOCHS, shuffle=True, verbose=1)
	del model2
	
	model3 = setModel(OriginalTrain, noise_level)
	history3 = model3.fit(x=NoisyTrain[int(NoisyTrain.shape[0]*0.5):int(NoisyTrain.shape[0]*0.75)], 
  						y=OriginalTrain[int(OriginalTrain.shape[0]*0.5):int(OriginalTrain.shape[0]*0.75)], 
  						batch_size=_BATCH_SIZE, epochs=_NUMBER_EPOCHS, shuffle=True, verbose=1)
	del model3
	
	model4 = setModel(OriginalTrain, noise_level)
	history4 = model4.fit(x=NoisyTrain[int(NoisyTrain.shape[0]*0.75):], 
  						y=OriginalTrain[int(OriginalTrain.shape[0]*0.75):], 
  						batch_size=_BATCH_SIZE, epochs=_NUMBER_EPOCHS, shuffle=True, verbose=1)
	del model4
	
	return history1, history2, history3, history4

#--------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	if not os.path.exists(_FOLDER_DATASET):
		os.makedirs(_FOLDER_DATASET)

	for scoreMeasure in _SCORING_METRIC:
		for noise_level in _NOISE_LEVEL:
			folderName = _FOLDER_MODELS+"/"+scoreMeasure+"_"+str(noise_level)+"/"
			if not os.path.exists(folderName):
				os.makedirs(folderName)
			if not os.path.exists(folderName+"/graphics/"):
				os.makedirs(folderName+"/graphics/")
			if not os.path.exists(folderName+"/scoring/"):
				os.makedirs(folderName+"/scoring/")

			if _DATASET in ["cifar", "mnist", "fashion_mnist", "bsds300", "fruits360"]:
				OriginalTrain, NoisyTrain, _, _ = loadData(noise_level, scoreMeasure, folderName+"/scoring/")
				trainingModel(OriginalTrain, NoisyTrain, "linear", folderName, noise_level)
				trainingModel(OriginalTrain, NoisyTrain, "ladder", folderName, noise_level)
				trainingModel(OriginalTrain, NoisyTrain, "log", folderName, noise_level)
				trainingModel(OriginalTrain, NoisyTrain, "ladderlog", folderName, noise_level)
				trainingModel(OriginalTrain, NoisyTrain, "constant", folderName, noise_level)
				history1, history2, history3, history4 = trainingHardnessScoring(OriginalTrain, NoisyTrain, noise_level)
				checkHardnessScoring(history1, history2, history3, history4, folderName+"/graphics/", _ANTI_CL)

				OriginalTest = np.load('%s%s_OriginalTest.npy' % (_FOLDER_DATASET, _DATASET))
				NoisyTest = np.load('%s%s_NoisyTest_%s.npy' % (_FOLDER_DATASET, _DATASET, str(noise_level)))
				predictingModel(OriginalTest, NoisyTest, "linear", folderName)
				predictingModel(OriginalTest, NoisyTest, "ladder", folderName)
				predictingModel(OriginalTest, NoisyTest, "log", folderName)
				predictingModel(OriginalTest, NoisyTest, "ladderlog", folderName)
				predictingModel(OriginalTest, NoisyTest, "constant", folderName)
				
				performanceGraphs(folderName+"/graphics/")

			else:
				OriginalTest, NoisyTest = loadDataOnlyTest(noise_level, scoreMeasure, folderName+"/scoring/")
				predictingModel(OriginalTest, NoisyTest, "linear", folderName.replace(_DATASET,_CROSS_DATASET))
				predictingModel(OriginalTest, NoisyTest, "ladder", folderName.replace(_DATASET,_CROSS_DATASET))
				predictingModel(OriginalTest, NoisyTest, "log", folderName.replace(_DATASET,_CROSS_DATASET))
				predictingModel(OriginalTest, NoisyTest, "ladderlog", folderName.replace(_DATASET,_CROSS_DATASET))
				predictingModel(OriginalTest, NoisyTest, "constant", folderName.replace(_DATASET,_CROSS_DATASET))
			
	print("\tDone!!")