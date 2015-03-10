import numpy as np
import cv2
import cPickle as pickle

def show (img, name="image"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return()

def trim (img):
    _,thresh = cv2.threshold(img,1,100,cv2.THRESH_BINARY)
    
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[np.argmax(map(len, contours))]
    x,y,w,h = cv2.boundingRect(cnt)
    
    crop = img[y:y+h,x:x+w]
    crop = cv2.resize(crop, (28,28), interpolation=cv2.INTER_LANCZOS4)
    return crop


def load ():
    with open('mnist.pkl', 'rb') as fp:
        trn, vld, tst = pickle.load(fp)
    for j in xrange(len(trn[0])):
        dig = trn[1][j]
        name = 'data/train/'+str(dig)+'/'+str(j)+'.png'
        img = trn[0][j].reshape((28,28))
        cv2.imwrite(name, trim((img*255).astype(np.uint8)))
    for j in xrange(len(vld[0])):
        dig = vld[1][j]
        name = 'data/valid/'+str(dig)+'/'+str(j)+'.png'
        img = vld[0][j].reshape((28,28))
        cv2.imwrite(name, trim((img*255).astype(np.uint8)))
    for j in xrange(len(tst[0])):
        dig = tst[1][j]
        name = 'data/test/'+str(dig)+'/'+str(j)+'.png'
        img = tst[0][j].reshape((28,28))
        cv2.imwrite(name, trim((img*255).astype(np.uint8)))

load()
