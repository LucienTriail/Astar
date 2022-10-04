import csv
import math
import numpy as np
import matplotlib.pyplot as plt

start = (0, 3)
end = (19, 20)
matrix = np.matrix([
    [1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1.],
    [1., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
    [1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
    [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1.],
    [1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1.],
    [1., 1., 1., 0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1.],
    [1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
    [1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1.],
    [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1.],
    [1., 0., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 1.],
    [1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1.],
    [1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 1.],
    [1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1.],
    [1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 0., 1.],
    [1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
    [1., 0., 1., 0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 0., 1.],
    [1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1.],
    [1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1.],
    [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0  # Cost
        self.h = 0  # Heuristic
        self.f = 0  # Total cost of present node

    def __eq__(self, other):
        return self.position == other.position


# matrix = np.matrix(
#  [[0., 0., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 1., 1., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 0., 0., 0.]])

def get_max_value_of_x_column(matrix, x):
    max = 0
    for row in matrix:
        if row[x] > max:
            max = row[x]
    return max


def csvToMatrix(csv):
    newMatrix = []
    csvData = open(csv)
    matrix = np.loadtxt(csvData, delimiter=";")
    matrixCartesien = []  # Matrice de conversion

    # Conversion des coordonnées polaires en cartésiennes
    for row in matrix:
        matrixCartesien.append(polaire_to_cartesien(row[1], row[0]))

    NEEDED = 2000
    CASE = 4
    STEP = NEEDED / CASE
    MIN = -2000
    MAX = 2000

    # On remplit la matrice de retour avec des 0
    for i in range(CASE):  # Axe des Y
        newMatrix.append([])
        for j in range(CASE):  # Axe X
            newMatrix[i].append(0.)

    for key, row in enumerate(matrixCartesien):
        x = row[0] + MAX
        y = row[1] + MAX
        if 0 < x < MAX and MIN < y < MAX:
            newMatrix[int(x / STEP)][int(y / STEP)] = 1.

    return newMatrix

def polaire_to_cartesien(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (x, y)


def astar(maze, start, end):
    # Start et end avec les valeurs initiales
    start_node = Node(None, tuple(start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, tuple(end))
    end_node.g = end_node.h = end_node.f = 0

    # tous les nœuds qui doivent encore être visités pour l'exploration
    yet_to_visit_list = []
    # tous les nœuds déjà visités
    visited_list = []

    yet_to_visit_list.append(start_node)

    outer_iterations = 0
    max_iterations = (len(maze) // 2) ** 10

    move = [
        [-1, 0],  # up
        [0, -1],  # left
        [1, 0],  # down
        [0, 1],  # right
    ]

    no_rows, no_columns = np.shape(maze)

    # Loop until we find the end
    while len(yet_to_visit_list) > 0:
        outer_iterations += 1

        current_node = yet_to_visit_list[0]
        current_index = 0
        for index, item in enumerate(yet_to_visit_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        if outer_iterations > max_iterations:
            return displayPath(maze, current_node)

        yet_to_visit_list.pop(current_index)
        visited_list.append(current_node)

        if current_node == end_node:
            good_positions = []
            while current_node.parent is not None:
                good_positions.append(current_node.position)
                current_node = current_node.parent
            good_positions.append(start_node.position)
            good_positions = good_positions[::-1]
            return good_positions

        # Generate children from all adjacent squares
        children = []

        for new_position in move:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
            if (node_position[0] > (no_rows - 1) or node_position[0] < 0 or node_position[1] > (no_columns - 1) or
                    node_position[1] < 0):
                continue

            if maze[node_position[0]][node_position[1]] != 0:
                continue

            new_node = Node(current_node, node_position)

            children.append(new_node)

        # Loop through children
        for child in children:
            if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                continue

            child.g = current_node.g + 1
            child.h = (((child.position[0] - end_node.position[0]) ** 2) +
                       ((child.position[1] - end_node.position[1]) ** 2))

            child.f = child.g + child.h

            if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                continue

            yet_to_visit_list.append(child)


def verify(maze, start, end, path):
    if path[0][0] != start[0] or path[0][1] != start[1] or path[-1][0] != end[0] or path[-1][1] != end[1]:
        print('Start or end incorrect')
        return False
    last = None
    for pos in path:
        if maze[pos] == 1:
            print('Path cross a wall')
            return False
        if last == None:
            last = pos
            continue
        diffX = abs(pos[0] - last[0])
        diffY = abs(pos[1] - last[1])
        if diffX > 1 or diffY > 1 or diffX + diffY > 1:
            print('Path not consecutive')
            return False
        last = pos
    print('Correct path')
    return True


def displayMatrix(mat):
    plt.matshow(mat)
    plt.show()

def displayPath(mat, path):
    if path != None:
        for (y, x) in path:
            mat[x, y] = math.inf
    plt.matshow(mat)
    plt.show()


newMatrix = csvToMatrix("data0.csv")
displayMatrix(newMatrix)
maze = np.asarray(newMatrix)
start = (0,0)
end = (3,3)

path = astar(maze, start, end)
displayPath(newMatrix, path)
print(verify(maze, start, end, path))
