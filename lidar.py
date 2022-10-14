import signal, sys

import numpy as np
from rplidar import RPLidar
from time import sleep
# from astar import *
from astar import get_lidar_data
from robot import moveDevice

LIDAR_PORT = '/dev/ttyUSB0'

matrix = []

def runRobot(maze, path, start, end) :
    # newPath = np.(arr, 2)
    newPath = []
    # # i  = 0
    newPath = []
    for i, pos in enumerate(path):
        if (i < 2):
            newPath.append(pos)
            # print(pos)
    print(newPath)
    moveDevice(newPath)
    maze[start] = -1.
    maze[end] = -1.
    print(maze)
    print(path)

while (True):
    print('starting...')
    sleep(2)
    try:
        lidar = RPLidar(LIDAR_PORT)
        for i, scan in enumerate(lidar.iter_scans()):
            for qual, angle, dist in scan:
                print(angle, dist)
                matrix.append((int(angle), int(dist)))
            break

        # print(matrix)
        # print(len(matrix))
        print('get_lidar_data()')
        (maze, path, start, end) = get_lidar_data(matrix)
        #print(maze, path, start, end)
        # runRobot(maze, path)
        #get_lidar_data(matrix)
        print('get DO')
        print(maze, path)
        runRobot(maze, path, start, end)
        # for (i, val) in enumerate(lidar.iter_measurments()):
        #     _, qual, angle, dist = val
        #     print(qual, angle, dist)
        #     break
        # lidar.stop()
        # lidar.stop_motor()
        # lidar.disconnect()
    except KeyboardInterrupt:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        break
    except SystemExit:
        break
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        try:
            print(str(exc_type) + '\t' + str(exc_value))
        except KeyboardInterrupt:
            lidar = RPLidar(LIDAR_PORT)
            lidar.stop()
            lidar.stop_motor()
            lidar.disconnect()



