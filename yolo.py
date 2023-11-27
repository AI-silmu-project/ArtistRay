from ultralytics import YOLO
import cv2
model = YOLO('best.pt')

capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = capture.read()
    
    if ret:
        frame = frame[300:700,600:1200,:]
        
        result = model.track(frame)
        for bbox in result[0].boxes:
            
            asdf = bbox.xyxy.numpy()[0]
            frame= cv2.rectangle(frame,(int(asdf[0]),int(asdf[1]),int(asdf[2]-asdf[0]),int(asdf[3]-asdf[1])),(0,0,255),2 )
        
        cv2.imshow("VideoFrame", frame)
        a = cv2.waitKey(1) 
        if a == ord('c'):
            cv2.imwrite('./cap.jpg',frame)
        elif a == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()