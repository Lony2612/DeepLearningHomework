import cv2
import numpy as np
import matplotlib.pyplot as plt

img =  cv2.imread('lena.jpg')



############## assigment 1 ##########
res = np.zeros((512,512,3))

res[:,:,0] = img[:,:,2]
res[:,:,1] = img[:,:,1]
res[:,:,2] = img[:,:,0]

# cv2.imwrite('result.jpg', res)

# cv2.imshow('img.jpg', res)
plt.imshow(img)
plt.show()
# cv2.waitKey()
# cv2.destroyAllWindows()

############### assigment 2 ############
## up
for ii in range(17,(18+90)):
    res[1,ii,0] = 0
    res[1,ii,1] = 0
    res[1,ii,2] = 255
## bottom
for ii in range(17,(18+90)):
    res[75,ii,0] = 0
    res[75,ii,1] = 0
    res[75,ii,2] = 255
## right
for ii in range(1,76):
    res[ii,17,0] = 0
    res[ii,17,1] = 0
    res[ii,17,2] = 255
## left
for ii in range(1,76):
    res[ii,107,0] = 0
    res[ii,107,1] = 0
    res[ii,107,2] = 255
############## assignment 3 ########
cv2.imwrite('result.jpg', res)

