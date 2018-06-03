import numpy as np
import cv2 as cv
import time

# local modules
from pynput.keyboard import Key, Controller 
from video import create_capture
from common import clock, draw_str

x1_a = 100
y1_a = 100
x2_a = 500
y2_a = 400

keyboard = Controller()

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)


def detect_intersect(rects):
    time.sleep(.01)
    for x1, y1, x2, y2 in rects:
        if y1_a > y1 & y2_a < y2 & x1_a < x1 & x2_a < x2:
            print("Center")
        elif y1_a > y1:
            keyboard.press(Key.up)
            print("Up")
        elif y2_a < y2:
            keyboard.press(Key.down)
            print("Down")
        elif x1_a > x1:
            keyboard.press(Key.right)
            print("Right")
        elif x2_a < x2:
            keyboard.press(Key.left)
            print("Left")
        

        
if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cascade_fn)
    nested = cv.CascadeClassifier(nested_fn)

    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')

    while True:
        ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        vis = cv.flip(img, 1) 
        draw_rects(vis, rects, (0, 255, 0))
        cv.rectangle(vis, (x1_a, y1_a), (x2_a, y2_a), (0,0,255), 2)
        dt = clock() - t

        detect_intersect(rects)
        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv.imshow('facedetect', vis)

        if cv.waitKey(5) == 27:
            break
    cv.destroyAllWindows()
