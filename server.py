from flask import Flask, make_response, jsonify
import numpy as np
from flask_cors import CORS
#from flask_compress import Compress
app = Flask(__name__)
#Compress(app)
CORS(app)
from glob import glob
from retrodetect import getblockmaxedimage
pathtoimgs = '/home/mike/Documents/Research/bee/photos_April5'

def getimgfilename(number):
    fns = sorted(glob('%s/*.np'%(pathtoimgs)))
    return fns[number]
    
@app.route('/')
def hello_world():
    return 'root node of bee label API.'

@app.route('/configure/<string:path>')
def configure(path):
    global pathtoimgs
    pathtoimgs = path
    return "set new path %s" % path

@app.route('/savepos/<int:number>/<int:x1>/<int:y1>/<int:x2>/<int:y2>')
def savepos(number,x1,y1,x2,y2):
    print("==========================")
    fn = getimgfilename(number)
    print(number,fn,(x2+x1)/2,(y2+y1)/2)
    with open("labels.csv", "a") as labelfile:
        labelfile.write("%d,%s,%d,%d\n" % (number,fn,int((x2+x1)/2),int((y2+y1)/2)))


    return "done"
    
@app.route('/getimage/<int:number>/<int:x1>/<int:y1>/<int:x2>/<int:y2>')
def getimage(number,x1,y1,x2,y2):
    global pathtoimgs
    print('%s/%04d'%(pathtoimgs,number))  
    print(x1,y1,x2,y2)
    #fns = sorted(glob('%s/*.np'%(pathtoimgs)))
    #if len(fns)==0:
    #    return "Image not found"
    fn = getimgfilename(number)
    n, img, data = np.load(fn,allow_pickle=True)
    if img is None:
        return ""
    print(img.shape)       
    steps = int((x2-x1)/500)
    if steps<1: steps = 1
    #img = (img.T[x1:x2:steps,y1:y2:steps]).T

    print(steps)
    img = (img.T[x1:x2,y1:y2]).T
    k = int(img.shape[0] / steps)
    l = int(img.shape[1] / steps)
    img = img[:k*steps,:l*steps].reshape(k,steps,l,steps).max(axis=(-1,-3))


    print(img.shape)
    img[int(img.shape[0]/2),:] = 255
    img[:,int(img.shape[1]/2)] = 255    
    return jsonify({'index':n,'photo':img.tolist(),'record':data})

if __name__ == "__main__":
    app.run(host="0.0.0.0")

