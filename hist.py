import cv2
import numpy
import glob

from matplotlib import pyplot as plt

images = [cv2.imread(file) for file in glob.glob("A7/znak/*.png")]
image_res = [cv2.resize(img, (32, 32)) for img in images]
h_images = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in image_res]

test = cv2.imread('2.png')
test_res = cv2.resize(test, (32, 32))
test_h = cv2.cvtColor(test_res, cv2.COLOR_BGR2HSV)
test_hist = cv2.calcHist(test_h, [0],None,[180],[0,180])



hist = cv2.calcHist(h_images[9], [0],None,[180],[0,180])
for i in range(8):
    hist += cv2.calcHist(h_images[i], [0],None,[180],[0,180])

print([i[0] for i in hist])
print(numpy.corrcoef([i[0] for i in hist], [i[0] for i in test_hist]))
plt.plot(hist*8)
plt.xlim([0,180])
plt.show()


# hist = cv2.calcHist(h_images[17], [1],None,[180],[0,180])
# for i in range(17):
#     hist += cv2.calcHist(h_images[i], [1],None,[180],[0,180])
# plt.plot(hist)
# plt.xlim([0,180])
# plt.show()

# hist = cv2.calcHist(h_images[17], [2],None,[180],[0,180])
# for i in range(17):
#     hist += cv2.calcHist(h_images[i], [2],None,[180],[0,180])
# plt.plot(hist)
# plt.xlim([0,180])
# plt.show()

