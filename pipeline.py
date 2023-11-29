from ultralytics import YOLO
import cv2
import numpy as np
import time
from segmentation.segmentation import segmentated_image
model = YOLO('best.pt')

FOCUS = 12
THRESH = 0.5
capture = cv2.VideoCapture(1,cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_AUTOFOCUS,0)
capture.set(cv2.CAP_PROP_FOCUS,FOCUS)

print('start')
last_fps_time = None
frame_count = 0
recorded_fps = 0
while True:
    ret, frame = capture.read()
    
    if ret:
        result = model.track(frame, conf=THRESH,verbose=False)
        '''
        for bbox in result[0].boxes:
            
            asdf = bbox.xyxy.numpy()[0]
            frame= cv2.rectangle(frame,(int(asdf[0]),int(asdf[1]),int(asdf[2]-asdf[0]),int(asdf[3]-asdf[1])),(0,0,255),2 )
        '''
        frame = result[0].plot()


        if last_fps_time is None:
            last_fps_time = time.time()
            frame_count=0
        elif time.time()-last_fps_time>=1.0:
            recorded_fps = frame_count/(time.time()-last_fps_time)
            frame_count = 0
            last_fps_time = time.time()
        frame_count+=1

        frame = cv2.putText(frame,f'{int(recorded_fps)} FPS',(0,14),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        frame = cv2.putText(frame,f'{FOCUS/255:.2f} Focus',(0,28),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        frame = cv2.putText(frame,f'{THRESH:.2f} Conf',(0,42),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        frame = cv2.putText(frame,'Press [C] to capture',(0,473),cv2.FONT_HERSHEY_SIMPLEX,0.5,(220,200,255),1,cv2.LINE_AA)
        cv2.imshow("Camera", frame)
        
        a = cv2.waitKey(1) 
        if a == ord('c'):
            for box in result[0].boxes:
                if box.id is not None:
                    print(box.id)
                    bbox = box.xyxy.numpy()[0].astype(np.int64)
                    cv2.imwrite('./cap.jpg',frame[bbox[1]:bbox[3],bbox[0]:bbox[2], :])
                    seg_result = segmentated_image(frame[bbox[1]:bbox[3],bbox[0]:bbox[2], :])
                    cv2.imwrite('./cap_seg.jpg',(seg_result*255)[:,:,None])
                    break
            
        elif a == ord('q'):
            break
    
        elif a == ord('w'):
            FOCUS+=3
            if FOCUS>255:
                FOCUS=255
            capture.set(cv2.CAP_PROP_FOCUS,FOCUS)
        elif a == ord('s'):
            FOCUS-=3
            if FOCUS<0:
                FOCUS=0
            capture.set(cv2.CAP_PROP_FOCUS,FOCUS)    
        elif a == ord('d'):
            THRESH+=0.05
            if THRESH>=0.75:
                THRESH=0.75
        elif a == ord('a'):
            THRESH-=0.05
            if THRESH<=0.1:
                THRESH=0.1


capture.release()
cv2.destroyAllWindows()