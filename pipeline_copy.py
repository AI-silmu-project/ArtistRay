
import cv2
import numpy as np
import time
import torch
import glob
import sys
import os
from types import SimpleNamespace

sys.path.append('SiamMask')
sys.path.append(os.path.join('SiamMask', 'experiments', 'siammask_sharp'))

from custom import Custom
from tools.test import *

# Camera Settings
FOCUS = 12
THRESH = 0.5

capture = cv2.VideoCapture(1,cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_AUTOFOCUS,0)
capture.set(cv2.CAP_PROP_FOCUS,FOCUS)
last_fps_time = None
frame_count = 0
recorded_fps = 0


if __name__ == "__main__":

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    # Setup Model

    cfg = load_config(SimpleNamespace(config='config_davis.json'))
    siammask = Custom(anchors=cfg['anchors'])
    siammask = load_pretrain(siammask, 'SiamMask_DAVIS.pth')

    siammask.eval().to(device)

    # Select ROI
    cv2.namedWindow("SiamMask")
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    

    selected = False
    while True:
        ret, frame = capture.read()
        
        if ret:

            if last_fps_time is None:
                last_fps_time = time.time()
                frame_count=0
            elif time.time()-last_fps_time>=1.0:
                recorded_fps = frame_count/(time.time()-last_fps_time)
                frame_count = 0
                last_fps_time = time.time()
            frame_count+=1

            if selected:
                state = siamese_track(state, frame, mask_enable=True, refine_enable=True, device=device)  # track
                location = state['ploygon'].flatten()
                mask = state['mask'] > state['p'].seg_thr

                frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
                cv2.polylines(frame, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                

            frame = cv2.putText(frame,f'{int(recorded_fps)} FPS',(0,14),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
            frame = cv2.putText(frame,f'{FOCUS/255:.2f} Focus',(0,28),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
            frame = cv2.putText(frame,f'{THRESH:.2f} Conf',(0,42),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
            frame = cv2.putText(frame,'Press [C] to capture',(0,473),cv2.FONT_HERSHEY_SIMPLEX,0.5,(220,200,255),1,cv2.LINE_AA)
            cv2.imshow("SiamMask", frame)
            
            a = cv2.waitKey(1) 
            if a == ord('c'):
                ret, frame = capture.read()
                try:
                    init_rect = cv2.selectROI('SiamMask', frame, False, False)
                    x, y, w, h = init_rect
                    selected = True
                    target_pos = np.array([x + w / 2, y + h / 2])
                    target_sz = np.array([w, h])
                    state = siamese_init(frame, target_pos, target_sz, siammask, cfg['hp'], device=device)
                except:
                    exit()
                
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