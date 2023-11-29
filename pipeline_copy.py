
import cv2
import numpy as np
import torch
import sys, os, time, subprocess
import skimage


# for importing SiamMask -------------------------------------------------------
from types import SimpleNamespace
sys.path.append('SiamMask')
sys.path.append(os.path.join('SiamMask', 'experiments', 'siammask_sharp'))
from custom import Custom
from tools.test import *
# ------------------------------------------------------------------------------


# Camera Settings --------------------------------------------------------------
H = 720
W = 1280
FOCUS = 0
EXPO = -6
capture = cv2.VideoCapture(1,cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
capture.set(cv2.CAP_PROP_FPS, 30)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, W)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, H)q
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE,0)
capture.set(cv2.CAP_PROP_AUTOFOCUS,0)
capture.set(cv2.CAP_PROP_AUTO_WB, 0)
capture.set(cv2.CAP_PROP_EXPOSURE,EXPO)
capture.set(cv2.CAP_PROP_FOCUS,FOCUS)
# ------------------------------------------------------------------------------

last_fps_time = None
frame_count = 0
recorded_fps = 0

def run_BLD(prompt: str, init_image_path: str, mask_path: str, output_path: str, batch_size: int = 3, ):
    print('Creating Images...')
    retcode = subprocess.call(['python', './blended-latent-diffusion/scripts/text_editing_stable_diffusion.py', '--prompt', prompt, '--init_image', init_image_path, '--mask', mask_path, '--output_path', output_path, '--batch_size', str(batch_size)])
    if retcode == 0:
        print(f'Image Creation Successful: {batch_size} images, {output_path}')
    return retcode

if __name__ == "__main__":

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    # Setup SiamMask
    cfg = load_config(SimpleNamespace(config='config_davis.json'))
    siammask = Custom(anchors=cfg['anchors'])
    siammask = load_pretrain(siammask, 'SiamMask_DAVIS.pth')
    siammask.eval().to(device)


    cv2.namedWindow("SiamMask",cv2.WINDOW_AUTOSIZE)

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
                cv2.polylines(frame, [np.intp(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                

            # information
            frame = cv2.putText(frame,f'{int(recorded_fps)} FPS',(0,14),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
            frame = cv2.putText(frame,f'{FOCUS/255:.2f} Focus',(0,28),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
            frame = cv2.putText(frame,f'{EXPO:.2f} Exposure',(0,42),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
            frame = cv2.putText(frame,'Press [C] to capture',(0,H-14),cv2.FONT_HERSHEY_DUPLEX,1,(220,200,255),2,cv2.LINE_AA)
            cv2.imshow("SiamMask", frame)
            
            # control keys
            a = cv2.waitKey(1) 
            if a == ord('c'): # capture and make images
                ret, frame = capture.read()
                init_rect = cv2.selectROI('SiamMask', frame, False, False)
                x, y, w, h = init_rect
                if w == 0 and h == 0:
                    continue
                selected = True
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                print('ROI Selection OK')

                state = siamese_init(frame, target_pos, target_sz, siammask, cfg['hp'], device=device)
                state = siamese_track(state, frame, mask_enable=True, refine_enable=True, device=device)
                print('Tracking Initialization OK')

                init_mask = (state['mask'] > state['p'].seg_thr) * 255

                cv2.imwrite('image.png', frame[y:y+h, x:x+w])
                cv2.imwrite('mask.png', init_mask.astype(np.uint8)[y:y+h, x:x+w])
                
                prompt = input('프롬프트를 입력해주세요: ')
                
                run_BLD(prompt,'./image.png','./mask.png','./outputs/output.png' )
            
            # focus control
            elif a == ord('s'):
                FOCUS+=15
                if FOCUS>255:
                    FOCUS=255
                capture.set(cv2.CAP_PROP_FOCUS,FOCUS)
            elif a == ord('w'):
                FOCUS-=15
                if FOCUS<0:
                    FOCUS=0
                capture.set(cv2.CAP_PROP_FOCUS,FOCUS)    
            
            # exposure control
            elif a == ord('d'):
                EXPO+=1
                if EXPO>=-2:
                    EXPO = -2
                capture.set(cv2.CAP_PROP_EXPOSURE,EXPO)
            elif a == ord('a'):
                EXPO-=1
                if EXPO<=-11:
                    EXPO = -11
                capture.set(cv2.CAP_PROP_EXPOSURE,EXPO)
            
            elif a == ord('q'):
                break


    capture.release()
    cv2.destroyAllWindows()