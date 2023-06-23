from enum import IntEnum
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import random

BLUE_CONE_URI = 'model://blue_cone'
YELLOW_CONE_URI = 'model://yellow_cone'
ORANGE_CONE_URI = 'model://big_cone'

DROPOUT_THRESHOLD = 0.85
COLOR_CHANGE_THRESHOLD = 0.95
SHIFT_RADIUS = 0.5 # meters

class Colors(IntEnum):
    BLUE = 0
    YELLOW = 1
    ORANGE = 2

class Cone:
    def __init__(self, x, y, z, color):
        self.x = x
        self.y = y
        self.z = z
        self.color = color

class SDFTrackParser:
    def __init__(self, sdf_file):
        self.filename = sdf_file
        self.blue_cones = []
        self.yellow_cones = []
        self.orange_cones = []
        self.shifted_cones = []
        self.in_order = []

    def parse_pose(self, text):
        splits = text.split(" ")
        return float(splits[0]), float(splits[1]), float(splits[2])
    
    def parse_file(self):
        root = ET.parse(self.filename).getroot()
        cones = root.findall('./model/include')
        for child in cones:
            """
                child[0] == 'pose'
                child[1] == 'uri' (i.e. color)
                child[2] == 'name' (i.e. cone number)
            """
            x, y, z = self.parse_pose(child[0].text)
            if child[1].text == BLUE_CONE_URI:
                cone = Cone(x, y, z, Colors.BLUE)
                self.blue_cones.append(cone)
                self.in_order.append(cone)
            elif child[1].text == YELLOW_CONE_URI:
                cone = Cone(x, y, z, Colors.YELLOW)
                self.yellow_cones.append(cone)
                self.in_order.append(cone)
            else:
                cone = Cone(x, y, z, Colors.ORANGE)
                self.yellow_cones.append(cone)
                self.in_order.append(cone)

    def graph_cones(self, show_shifted=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.ax_graph(ax, self.in_order)
        if show_shifted:
            self.ax_graph(ax, self.shifted_cones, shifted_cones=True)
        ax.set(xlabel=None, ylabel=None, zlabel=None)
        plt.axis('off')
        plt.show()

    def ax_graph(self, ax, cones, shifted_cones=False):
        blue_x = [cone.x for cone in cones if cone.color == Colors.BLUE]
        blue_y = [cone.y for cone in cones if cone.color == Colors.BLUE]
        blue_z = [cone.z for cone in cones if cone.color == Colors.BLUE]

        yellow_x = [cone.x for cone in cones if cone.color == Colors.YELLOW]
        yellow_y = [cone.y for cone in cones if cone.color == Colors.YELLOW]
        yellow_z = [cone.z for cone in cones if cone.color == Colors.YELLOW]
        
        ax.scatter(blue_x, blue_y, blue_z, color='blue' if shifted_cones else 'grey')
        ax.scatter(yellow_x, yellow_y, yellow_z, color='yellow' if shifted_cones else 'grey')

    def generate_shifted_cones(self):
        self.shifted_cones = []
        for cone in self.in_order:
            if random.uniform(0, 1) < DROPOUT_THRESHOLD:
                new_x = cone.x + random.uniform(-1*SHIFT_RADIUS, SHIFT_RADIUS)
                new_y = cone.y + random.uniform(-1*SHIFT_RADIUS, SHIFT_RADIUS)
                new_color = cone.color if random.uniform(0, 1) < COLOR_CHANGE_THRESHOLD else Colors((int(cone.color) + 1)%3)
                self.shifted_cones.append(Cone(new_x, new_y, cone.z, new_color))

parser = SDFTrackParser('model.sdf')
parser.parse_file()
parser.graph_cones()
parser.generate_shifted_cones()
parser.graph_cones(show_shifted=True)
