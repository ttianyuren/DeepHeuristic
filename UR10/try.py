import pybullet as p
import time
import math
import pybullet_data


def fun(x, y, h=0.9):
    print(x + y + h)


xy = [0.1, 0.2]
fun(*xy, h=0.7)
