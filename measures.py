import json
import numpy as np

from statistics import mean
from skimage.metrics import mean_squared_error as rmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# graphics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------MEASURES-------------------------------------------------------------------------------

def mean_measures(OriginalTest, prediction):
  var_rmse = []
  var_psnr = []
  var_ssim = []

  for i in range(0, OriginalTest.shape[0]):
    var_rmse.append(rmse(OriginalTest[i], prediction[i]))    
    var_psnr.append(psnr(OriginalTest[i], prediction[i]))

    imgOriginal = np.reshape(OriginalTest[i], OriginalTest[i].shape[0] * OriginalTest[i].shape[1] * OriginalTest[i].shape[2])
    imgPrediction = np.reshape(prediction[i], prediction[i].shape[0] * prediction[i].shape[1] * prediction[i].shape[2])
    var_ssim.append(ssim(imgOriginal, imgPrediction, data_range=imgPrediction.max() - imgPrediction.min()))

  return mean(var_rmse), mean(var_psnr), mean(var_ssim)

#-------------------------------------------------------------------------GRAPHICS-------------------------------------------------------------------------------

def checkHardnessScoring(history1, history2, history3, history4, folderName, _ANTI_CL):
  print("\t\tPlotting the training loss graphic ...")
  if not _ANTI_CL:
    plt.plot(history1.history['loss'], label="Easier")
    plt.plot(history2.history['loss'], label="Easy")
    plt.plot(history3.history['loss'], label="Hard")
    plt.plot(history4.history['loss'], label="Harder")
  else:
    plt.plot(history4.history['loss'], label="Easier")
    plt.plot(history3.history['loss'], label="Easy")
    plt.plot(history2.history['loss'], label="Hard")
    plt.plot(history1.history['loss'], label="Harder")

  plt.legend(loc='upper right')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.xlim(left=1)
  plt.ylim(bottom=0)
  plt.margins(0.5, 0.5)
  plt.savefig("%shardness.png" % folderName)
  plt.close()


def performanceGraphs(folderName):
  print("\tPlotting the loss performance ...")
  with open('%s%s_loss.txt' % (folderName.replace("graphics/",""), "constant")) as f:
    constantLoss = json.load(f)
  with open('%s%s_loss.txt' % (folderName.replace("graphics/",""), "linear")) as f:
    linearLoss = json.load(f)
  with open('%s%s_loss.txt' % (folderName.replace("graphics/",""), "log")) as f:
    logLoss = json.load(f)
  with open('%s%s_loss.txt' % (folderName.replace("graphics/",""), "ladder")) as f:
    ladderLoss = json.load(f)
  with open('%s%s_loss.txt' % (folderName.replace("graphics/",""), "ladderlog")) as f:
    ladderlogLoss = json.load(f)
  generateGraphs(constantLoss, linearLoss, logLoss, ladderLoss, ladderlogLoss, "%sloss.png" % folderName, 'Loss', 'upper right')

  print("\tPlotting the example size performance ...")
  with open('%s%s_size.txt' % (folderName.replace("graphics/",""), "constant")) as f:
    constantSize = json.load(f)
  with open('%s%s_size.txt' % (folderName.replace("graphics/",""), "linear")) as f:
    linearSize = json.load(f)
  with open('%s%s_size.txt' % (folderName.replace("graphics/",""), "log")) as f:
    logSize = json.load(f)
  with open('%s%s_size.txt' % (folderName.replace("graphics/",""), "ladder")) as f:
    ladderSize = json.load(f)
  with open('%s%s_size.txt' % (folderName.replace("graphics/",""), "ladderlog")) as f:
    ladderlogSize = json.load(f)
  generateGraphs(constantSize, linearSize, logSize, ladderSize, ladderlogSize, "%ssize.png" % folderName, 'Number of examples', 'lower right')


def generateGraphs(constant, linear, log, ladder, ladderlog, graphName, measure, loc):
  plt.plot(constant, label="Constant")
  plt.plot(linear, label="Linear")
  plt.plot(log, label="Log")
  plt.plot(ladder, label="Ladder")
  plt.plot(ladderlog, label="Ladder Log")
  plt.legend(loc=loc)
  plt.ylabel(measure)
  plt.xlabel('Epochs')
  plt.xlim(left=1)
  plt.ylim(bottom=0)
  plt.margins(0.5, 0.5)
  plt.savefig(graphName)
  plt.close()