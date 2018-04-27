import socket
from datetime import datetime
from time import sleep

port = 5656
server_ip = "192.168.12.49"

def Client(host,port):
    s=socket.socket()
    s.connect((host,port))

    while True:
        s.send("hi")
        print host,s.recv(4096)


def Server(port):
    s = socket.socket()
    s.bind(('', port))

    while True:
        print('listening')
        s.listen(5)
        c, addr = s.accept()
        print('receiving')
        print(c.recv(4096))
        while True:
            print('sending')
            now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            try:
                c.send(now)
            except:
                break
            sleep(1)
        c.close()
    s.close()



if __name__=='__main__':
    #Server(5656)
    Client(server_ip,port)
