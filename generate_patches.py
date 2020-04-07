import numpy as np
from numpy import unravel_index
from glob import glob
pathtoimgs = '/home/mike/Documents/Research/bee/photos_April5'
import pandas as pd
import png
from retrodetect import getblockmaxedimage

labels = pd.read_csv('labels.csv',header=None)
patchn = 0
savesize = 40
delsize=15

def getimgfilename(number):
    fns = sorted(glob('%s/*.np'%(pathtoimgs)))
    return fns[number]
    
for l in labels.to_numpy():

    n =l[0]
    fn=l[1]
    truex =l[2]
    truey =l[3]
    
    _, img, data = np.load(fn,allow_pickle=True)
    _, lastimg, lastdata = np.load(getimgfilename(n-1),allow_pickle=True)
#    _, nextimg, nextdata = np.load(getimgfilename(n+1),allow_pickle=True)
    print(img,lastimg)
    if lastimg is None:
        _, lastimg, nextdata = np.load(getimgfilename(n+1),allow_pickle=True)
    img = np.array(img).astype(np.float) - getblockmaxedimage(np.array(lastimg).astype(np.float),5,3)
    img[img<0]=0
    img = img.astype(np.uint8)
    print(img)
    searchimg = img.copy()
    print(fn,n,truex,truey)
    
    
    for i in range(10):
        y,x = unravel_index(searchimg.argmax(), searchimg.shape)
        if (x<savesize) or (y<savesize) or (x>searchimg.shape[1]-savesize-1) or (y>searchimg.shape[0]-savesize-1): continue
        target = 1*(((y-truey)**2 + (x-truex)**2)<10**2)
        patch = 3*img[y-savesize:y+savesize,x-savesize:x+savesize].astype(np.float32)
        
        patch[patch>255] = 255
        if target:
            patch[2:-2,2]=255
            patch[2,2:-2]=255
            patch[2:-2,-2]=255        
            patch[-2,2:-2]=255        
        patch = patch.astype(np.uint8)
        png.from_array(patch,'L').save("patch%04d_%d.png" % (patchn,target))    
        patchn+=1
        searchimg[y-delsize:y+delsize,x-delsize:x+delsize]=0
    

