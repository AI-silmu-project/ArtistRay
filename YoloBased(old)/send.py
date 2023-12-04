import cv2
import socket

IP = '127.0.0.1'
PORT = 9505

W, H = 1280, 720

sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FPS, 30)
SIZE = H*W*3
SEC = SIZE // 10240
PACKSIZE = SIZE//SEC
while True:
    ret, frame = cap.read()
    data = frame.flatten()
    dumped = data.tobytes()
    for i in range(SEC):
        sock.sendto(bytes([i//256])+bytes([i%256])+dumped[i*PACKSIZE : (i+1)*PACKSIZE], (IP,PORT))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
