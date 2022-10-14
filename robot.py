# à lancer sur le robot
import serial
from time import sleep
import os


robotPort = os.popen('ls /dev/ttyACM*').read().strip()
print("Found serial : " + robotPort)

def openSerial():
    return serial.Serial(robotPort)

def sendInstruction(char, ser):
    ser.write(char.encode())

def moveDevice(path):

    last = None
    ser = openSerial()

    for pos in path:


        if last == None:
            last = pos
            continue

        if pos[0] != last[0] or pos[1] != last[1]:
            diffY = pos[0] - last[0]
            diffX = pos[1] - last[1]

            if diffX > 0:
                move('z', ser)
                print(diffX, ' avance')


            if diffY > 0:
                move('d', ser)
                print(diffY, ' à droite')
            else:
                if diffY < 0:
                    move('q', ser)
                    print(diffY, ' à gauche')



        else:
            # for i in range(5):
            #     sleep(1 / (i + 1))
            #     sendInstruction('z', ser)

            print('nothing')

        last = pos

    ser.close()



def move(cmd, ser):
    for i in range(8):
        sleep(0.1)
        sendInstruction(cmd, ser)


