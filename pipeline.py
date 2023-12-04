
import cv2
import numpy as np
import torch
import sys, os, time, subprocess


# for importing SiamMask -------------------------------------------------------
from types import SimpleNamespace
sys.path.append('SiamMask')
sys.path.append(os.path.join('SiamMask', 'experiments', 'siammask_sharp'))
from custom import Custom
from tools.test import *
# ------------------------------------------------------------------------------

CAM_ID = 0
if len(sys.argv) == 2:
    CAM_ID= int(sys.argv[1])


# Camera Settings --------------------------------------------------------------
H = 720
W = 1280
CROP = 512
FOCUS = 0
EXPO = -7
capture = cv2.VideoCapture(CAM_ID,cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
capture.set(cv2.CAP_PROP_FPS, 30)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, W)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE,0)
capture.set(cv2.CAP_PROP_AUTOFOCUS,0)
capture.set(cv2.CAP_PROP_AUTO_WB, 0)
capture.set(cv2.CAP_PROP_EXPOSURE,EXPO)
capture.set(cv2.CAP_PROP_FOCUS,FOCUS)
# ------------------------------------------------------------------------------

last_fps_time = None
frame_count = 0
recorded_fps = 0
opacity = 0.8

def run_BLD(prompt: str, init_image_path: str, mask_path: str, output_path: str, batch_size: int = 3, ):
    print('Creating Images...')
    retcode = subprocess.call(['python', './blended-latent-diffusion/scripts/text_editing_stable_diffusion.py', '--prompt', prompt, '--init_image', init_image_path, '--mask', mask_path, '--output_path', output_path, '--batch_size', str(batch_size)])
    if retcode == 0:
        print(f'Image Creation Successful: {batch_size} images, {output_path}')
    return retcode

class LowPassFilter(object):
    def __init__(self, cut_off_freqency=5, ts= 1./20.):
        self.ts = ts
        self.cut_off_freqency = cut_off_freqency
        self.tau = self.get_tau()

        self.prev_data = 0.
        
    def get_tau(self):
        return 1 / (2 * np.pi * self.cut_off_freqency)

    def filter(self, data):
        val = (self.ts * data + self.tau * self.prev_data) / (self.tau + self.ts)
        self.prev_data = val
        return val

if __name__ == "__main__":

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    # Setup SiamMask
    cfg = load_config(SimpleNamespace(config='config/config_davis.json'))
    siammask = Custom(anchors=cfg['anchors'])
    siammask = load_pretrain(siammask, 'config/SiamMask_DAVIS.pth')
    siammask.eval().to(device)

    lpf = LowPassFilter()

    cv2.namedWindow("SiamMask", cv2.WINDOW_AUTOSIZE)

    selected = False
    set_image = None
    to_paste = None
    ref_crop = None
    ref_frame = None

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
                
                center = np.intp(location.reshape((-1, 2)).mean(0))
                
                if ref_frame is not None:
                    loca = location.reshape(-1, 2)

                    cnt = location.reshape((-1, 1, 2))
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    height,width = to_paste.shape[:2]
                    src_pts = box.astype(np.float32)
                    dst_pts = np.array([[0, 0],
                                        [width-1, 0],
                                        [width-1, height-1],
                                        [0, height-1],], dtype=np.float32)
                    M = cv2.getPerspectiveTransform(dst_pts, src_pts)
                    M = lpf.filter(M)
                    warped2 = cv2.warpPerspective(to_paste, M, (W,H),borderMode=cv2.BORDER_TRANSPARENT)
                    
                    np.copyto(frame, (frame*(1.-opacity) + warped2[:,:,:3]*opacity).astype(np.uint8), where=(np.stack([warped2[:,:,3]]*3,-1)>0) )
                else:
                    frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
                    cv2.polylines(frame, [np.intp(location).reshape((-1, 1, 2))], False, (150, 0, 0), 3)

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

                
                if target_pos[1]<= CROP/2:
                    ymin = 0
                    ymax = CROP
                elif target_pos[1]>= H-CROP//2:
                    ymax = H
                    ymin = int(H-CROP)
                else:
                    ymin = int(target_pos[1] - CROP/2)
                    ymax = int(target_pos[1] + CROP/2)
                if target_pos[0]<= CROP/2:
                    xmin = 0
                    xmax = CROP
                elif target_pos[0]>= W-CROP/2:
                    xmax = W
                    xmin = int(W-CROP)
                else:
                    xmin = int(target_pos[0] - CROP/2)
                    xmax = int(target_pos[0] + CROP/2)

                cv2.imwrite('outputs/image.png', frame)
                cv2.imwrite('outputs/image_roi.png', frame[y:y+h, x:x+w])
                cv2.imwrite('outputs/mask.png', init_mask.astype(np.uint8))
                cv2.imwrite('outputs/mask_roi.png', init_mask.astype(np.uint8)[y:y+h, x:x+w])

                

                # BLD
                bld_mask = init_mask[ymin:ymax, xmin:xmax]
                assert bld_mask.shape == (CROP,CROP)
                cv2.imwrite('outputs/image_bldin.png', frame[ymin:ymax, xmin:xmax])
                cv2.imwrite('outputs/mask_bldin.png', bld_mask)
                prompt = input('프롬프트를 입력해주세요: ')
                run_BLD(prompt,'outputs/image_bldin.png','outputs/mask_bldin.png','outputs/image_bldout.png' , 1)
                
                bld_image = cv2.resize(cv2.imread('outputs/image_bldout.png'),(CROP,CROP))
                ref_frame = np.copy(frame)
                frame[ymin:ymax, xmin:xmax] = bld_image
                set_image = cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
                set_image[:,:, 3] = init_mask
                
                # 이제 frame에 그게 덧씌워짐
                # set_image는 투명하게 마스킹된거
                cv2.imwrite('outputs/output_overlay.png',frame)
                cv2.imwrite('outputs/output_mask.png', set_image)
                cv2.imwrite('outputs/output_roi.png', set_image[y:y+h, x:x+w, :])
                

                cnt = state['ploygon'].flatten().reshape((-1, 1, 2))
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                width = int(rect[1][1])
                height = int(rect[1][0])
                src_pts = box.astype("float32")
                dst_pts = np.array([[0, 0],
                                    [width-1, 0],
                                    [width-1, height-1],
                                    [0, height-1],], dtype="float32")
                M_init = cv2.getPerspectiveTransform(src_pts, dst_pts)
                to_paste = cv2.warpPerspective(set_image, M_init, (width,height))
            elif ref_frame is not None and a == ord('r'):
                pass
                
            # 초점
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
            
            # 노출
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