import cv2 
import os

img_patokan = cv2.imread('images/lena.jpg', 0)
# img_patokan = cv2.equalizeHist(img_patokan)
img_patokan = cv2.GaussianBlur(img_patokan, (5,5), cv2.BORDER_DEFAULT)
img_scene = []

# isi img_scene
for i in os.listdir('images'):
    if i == 'lena.jpg':
        continue
    else:
        gray = cv2.imread('images/'+i, 0)
        # gray = cv2.equalizeHist(gray)
        img_scene.append(gray)
count = 0
img_res = []

for curr_img in img_scene:
    # surf -> keypoint sama descriptor
    surf = cv2.xfeatures2d.SURF_create()
    # detect and compute -> kp, desc
    kp_patokan, des_patokan = surf.detectAndCompute(img_patokan, None)
    kp_scene, des_scene = surf.detectAndCompute(curr_img, None)
    # Buat flann -> cari matches
    flann = cv2.FlannBasedMatcher(dict(algorithm = 0))
    # ubah descriptor -> float, biar lebih akurat
    des_patokan = des_patokan.astype('f')
    des_scene = des_scene.astype('f')
    # process matches
    matches = flann.knnMatch(des_patokan, des_scene, 2)
    # process valid_match
    valid_match = []
    for i in range(len(matches)):
        valid_match.append([0,0])
    tempCount = 0
    # Lowe
    for i, (p,q) in enumerate(matches):
        if p.distance < 0.7 * q.distance:
            valid_match[i] = [1,0]
            tempCount += 1
    if tempCount > count:
        count = tempCount
        img_res = cv2.drawMatchesKnn(img_patokan, kp_patokan, curr_img, kp_scene, matches, None, matchColor = [0,0,255], singlePointColor = [0,255,0], matchesMask = valid_match)

cv2.imshow('Image Result', img_res)
cv2.waitKey(0)
cv2.destroyAllWindows()
