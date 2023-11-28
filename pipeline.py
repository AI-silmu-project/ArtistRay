from ultralytics import YOLO
import cv2
import numpy as np
from segmentation.segmentation import segmentated_image
model = YOLO('best.pt')

capture = cv2.VideoCapture(1,cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = capture.read()
    
    if ret:
        frame = frame
        
        result = model.track(frame)
        '''
        for bbox in result[0].boxes:
            
            asdf = bbox.xyxy.numpy()[0]
            frame= cv2.rectangle(frame,(int(asdf[0]),int(asdf[1]),int(asdf[2]-asdf[0]),int(asdf[3]-asdf[1])),(0,0,255),2 )
        '''

        cv2.imshow("VideoFrame", result[0].plot())
        
        a = cv2.waitKey(1) 
        if a == ord('c'):
            for box in result[0].boxes:
                if box.id is not None:
                    print(box.id)
                    bbox = box.xyxy.numpy()[0].astype(np.int64)
                    cv2.imwrite('./cap.jpg',frame[bbox[1]:bbox[3],bbox[0]:bbox[2], :])
                    seg_result = segmentated_image(frame[bbox[1]:bbox[3],bbox[0]:bbox[2], :])
                    print(seg_result)
                    cv2.imwrite('./cap_seg.jpg',(seg_result*255)[:,:,None])
                    break
            
        elif a == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()