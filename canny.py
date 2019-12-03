import matplotlib.image as img
import matplotlib.pyplot as plt 
import numpy as np 
from numpy import unravel_index
import cv2
from scipy import ndimage

# Read image and convert to grayscale
lena = img.imread('lena.png')
plt.figure()
plt.imshow(lena)
plt.title('Original image')

# Convert to grayscale
lena = cv2.cvtColor(lena, cv2.COLOR_RGB2GRAY)

# check to see if the cv normalizes the grayscaled
# image and compare to matlab value
#print(np.amax(lena))
#print(unravel_index(lena.argmax(), lena.shape))

plt.figure()
plt.imshow(lena, cmap="gray")
plt.title('Grayscaled image')

# Create Gaussian filter
kernel_row = 7
kernel_col = 7
sigma = 2
a = np.arange(-(kernel_row - 1)/2, 1 + (kernel_row - 1)/2)
b = np.arange(-(kernel_col - 1)/2, 1 + (kernel_col - 1)/2)
x, y = np.meshgrid(a, b)
hg = np.exp(-(x**2 + y**2) / (2 * sigma**2))
h = np.divide(hg, np.sum(hg))
# apply the Gaussian blur
blur = ndimage.filters.convolve(lena, h)
plt.figure()
plt.imshow(blur, cmap='gray')
plt.title('Smoothing the image using a %iX%i Gaussian '\
	'filter with sigma = %i' %(kernel_row, kernel_col, sigma))

# Doing the samething using opencv's GaussianBlur
#blur2 = cv2.GaussianBlur(lena, (7, 7), 2)
#plt.figure()
#plt.imshow(blur2)
#plt.title('Smoothing using opencv\'s GaussianBlur')

# Create the Sobel operator and apply it to the smoothed
# image
Sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
# compute the gradients in x and y directions
Gx = ndimage.filters.convolve(blur, Sx)
Gy = ndimage.filters.convolve(blur, Sy)
# show the gradients in both directions
plt.figure()
plt.imshow((Gx + 4)/8, cmap='gray')
plt.title('Gradient in the x-direction')
plt.figure()
plt.imshow((Gy + 4)/8, cmap='gray')
plt.title('Gradient in the y-direction')
# compute the gradient magnitude
Gmag = np.hypot(Gx, Gy)
plt.figure()
plt.imshow(Gmag / (4 * np.sqrt(2)), cmap='gray')
plt.title('Gradient magnitude')
# compute the gradient direction
theta = np.arctan2(-Gy, Gx)
# show the gradient directions
plt.figure()
plt.imshow((theta + np.pi) / (2 * np.pi), cmap='gray')
plt.title('Gradient directions')

# Non-maximum suppression
Y, X = lena.shape
Z = np.zeros((Y, X))
# convert the gradient directions to degrees
angle = theta * 180 / np.pi
angle[angle < 0] += 180
# check the direction of the gradient
before = 0
after = 0
for y in range(1, (Y - 1)):
	for x in range(1, (X - 1)):
		# degree = 0
		if (0 <= angle[y, x] < 22.5) or (157.5 <= angle[y, x] <= 180):
			before = Gmag[y, x-1]
			after = Gmag[y, x+1]
		# degree = 45
		elif (22.5 <= angle[y, x] < 67.5):
			before = Gmag[y+1, x-1]
			after = Gmag[y-1, x+1]
		# degree = 90
		elif (67.5 <= angle[y, x] < 112.5):
			before = Gmag[y-1, x]
			after = Gmag[y+1, x]
		# degree = 135
		elif(112.5 <= angle[y, x] < 157.5):
			before = Gmag[y-1, x-1]
			after = Gmag[y+1, x+1]
		# keep the brightest pixel and suppress others
		if (Gmag[y, x] >= before) and (Gmag[y, x] >= after):
			Z[y, x] = Gmag[y, x]
		else:
			Z[y, x] = 0
# plot image with non-maxima suppression applied
plt.figure()
plt.imshow(Z / (4 * np.sqrt(2)), cmap='gray')
plt.title('Gradient with non-maxima suppression applied')

# Double threshold
lowThresholdRatio = 0.05
highThresholdRatio = 0.09
highThreshold = Z.max() * highThresholdRatio
lowThreshold = highThreshold * lowThresholdRatio
T = np.zeros((Y, X))
weak = 25
strong = 255
strongY, strongX = np.where(Z >= highThreshold)
zerosY, zerosX = np.where(Z < lowThreshold)
weakY, weakX = np.where((Z < highThreshold) & (Z > lowThreshold))
T[strongY, strongX] = strong
T[weakY, weakX] = weak
# plot image with double threshold applied
plt.figure()
plt.imshow(T / (4 * np.sqrt(2)), cmap='gray')
plt.title('Double thresholding')


# show images
plt.show()