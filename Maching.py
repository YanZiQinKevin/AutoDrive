import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits=datasets.load_digits()
clf=svm.SVC(gamma=0.001,C=100)

clf.fit(digits.data[:-10],digits.target[:-10])
#print ('Prediction:', clf.predict(digits.data[-1]))

plt.imshow(digits.images[-2],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()
