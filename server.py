import socket, cv2
import numpy as np
IP = '127.0.0.1'
PORT = 9505
H, W = 720, 1280
SIZE = H*W*3
SEC = SIZE // 10240
PACKSIZE = SIZE//SEC

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((IP, PORT))
s = [[b'\xff' * PACKSIZE] for _ in range(SEC)]

while True:
    pic = b''
    data, addr = sock.recvfrom(PACKSIZE+2)
    idx = int(data[0]) *256 + int(data[1])
    s[idx] = data[2:PACKSIZE+2]
    if(idx == SEC-1):
        for i in range(SEC):
            pic += s[i]

        f = np.frombuffer(pic, dtype=np.uint8)
        f = f.reshape((H,W,3))
        cv2.imshow('frame',f)
        pic=b''

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break 
sock.close()