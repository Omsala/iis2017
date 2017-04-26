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


# 0.2 Dimensionality reduction using PCA
from sklearn.decomposition import PCA
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
pca = PCA(n_components=2)
X_trans = pca.fit_transform(data)
plt.scatter(X_trans[:,0], X_trans[:,1])
plt.show()

# 0.3 Mainfold embedding with tSNE
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_trans = tsne.fit_transform(data)
plt.scatter(X_trans[:,0], X_trans[:,1])
plt.show()

# 0.4 Classifer training and evaluation using k-NN
fnData = data
fnDigits = digits
split = 0.7
percentSplit = split
nSamples = n_samples
def holdOut(fnDigits,fnData,nSamples,percentSplit=0.8):
    n_trainSamples = int(nSamples*percentSplit)
    trainData = fnData[:n_trainSamples,:]
    trainLabels = fnDigits.target[:n_trainSamples]
    testData = fnData[n_trainSamples:,:]
    expectedLabels = fnDigits.target[n_trainSamples:]
    return trainData, trainLabels, testData, expectedLabels

trainData, trainLabels, testData, expectedLabels = holdOut(fnDigits, fnData, nSamples, percentSplit)


from sklearn import neighbors
n_neighbors = 10
kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
kNNClassifier.fit(trainData, trainLabels)
predictedLabels = kNNClassifier.predict(testData)
print("Classification report for classifier %s: \n %s \n"
% ('k-NearestNeighbour', metrics.classification_report(expectedLabels, predictedLabels)))
print("Confusion matrix:\n %s" % metrics.confusion_matrix(expectedLabels, predictedLabels))

#trainData, trainLabels, testData, expectedLabels = cross_validation.train_test_split(fnData, fnDigits.target,test_size=(1.0-percentSplit))
kFold = 10
scores = cross_validation.cross_val_score(kNNClassifier, fnData, fnDigits.target, cv=kFold)
print(scores)


# 0.5 Classifer training and evaluation using SVM
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
clf_svm = LinearSVC()
clf_svm.fit(trainData, trainLabels)
predictedLabels = clf_svm.predict(testData)
acc_svm = accuracy_score(expectedLabels, predictedLabels)
print "Linear SVM accuracy: ", acc_svm

# Display classification results
kFold = 10
scores = cross_validation.cross_val_score(clf_svm, fnData, fnDigits.target, cv=kFold)
print(scores)

