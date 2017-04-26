from sklearn import datasets
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn import metrics
from sklearn import cross_validation

digits = datasets.load_digits()

# 0.1 Data Visualization
i = 0
for image in digits.images:
    if(i < 10):
        imMax = np.max(image)
        image = 255*(np.abs(imMax-image)/imMax)
        res = cv2.resize(image,(100, 100), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite('digit'+str(i)+'.png',res)
        i+=1
    else:
        break

raw_input("Press enter to continue...")

# 0.2 Dimensionality reduction using PCA
from sklearn.decomposition import PCA
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
pca = PCA(n_components=2)
X_trans = pca.fit_transform(data)
plt.scatter(X_trans[:,0], X_trans[:,1])
plt.show()

raw_input("Press enter to continue...")

# 0.3 Mainfold embedding with tSNE

raw_input("Press enter to continue...")

# 0.4 Classifer training and evaluation using k-NN
fnData = data
fnDigits = digits
split = 0.7
percentSplit = split
nSamples = n_samples
def holdOut(fnDigits,fnData,nSamples,percentSplit=0.8):
    pass

trainData, trainLabels, testData, expectedLabels = holdOut(fnDigits, fnData, nSamples, percentSplit)

raw_input("Press enter to continue...")

# 0.5 Classifer training and evaluation using SVM


