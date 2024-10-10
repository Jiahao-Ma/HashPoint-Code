import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tools import Point


class Node:
    def __init__(self, x, y, width, height, capacity):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.capacity = capacity
        self.points = []
        self.divided = False
        self.northeast = None
        self.northwest = None
        self.southeast = None
        self.southwest = None

    def contains(self, point):
        return (point.position[0] >= self.x - self.width / 2 and
                point.position[0] < self.x + self.width / 2 and
                point.position[1] >= self.y - self.height / 2 and
                point.position[1] < self.y + self.height / 2)

    def subdivide(self):
        hw = self.width / 2
        hh = self.height / 2
        self.northeast = Node(self.x + hw / 2, self.y + hh / 2, hw, hh, self.capacity)
        self.northwest = Node(self.x - hw / 2, self.y + hh / 2, hw, hh, self.capacity)
        self.southeast = Node(self.x + hw / 2, self.y - hh / 2, hw, hh, self.capacity)
        self.southwest = Node(self.x - hw / 2, self.y - hh / 2, hw, hh, self.capacity)
        self.divided = True

    # def insert(self, point):
    #     if not self.contains(point):
    #         return False

    #     if len(self.points) < self.capacity:
    #         self.points.append(point)
    #         return True

    #     if not self.divided:
    #         self.subdivide()

    #     return (self.northeast.insert(point) or
    #             self.northwest.insert(point) or
    #             self.southeast.insert(point) or
    #             self.southwest.insert(point))
    def insert(self, point):
        if not self.contains(point):
            return False

        if not self.divided:
            if len(self.points) < self.capacity:
                self.points.append(point)
                return True
            else:
                self.subdivide()
                for existing_point in self.points:
                    self.northeast.insert(existing_point) or \
                    self.northwest.insert(existing_point) or \
                    self.southeast.insert(existing_point) or \
                    self.southwest.insert(existing_point)
                self.points = [] 
        
        return (self.northeast.insert(point) or
                self.northwest.insert(point) or
                self.southeast.insert(point) or
                self.southwest.insert(point))


    def show(self, ax):
        ax.add_patch(patches.Rectangle(
            (self.x - self.width / 2, self.y - self.height / 2),
            self.width, self.height, fill=False, edgecolor="black"
        ))

        # for point in self.points:
        #     ax.plot(point.x, point.y, 'r.')

        if self.divided:
            self.northeast.show(ax)
            self.northwest.show(ax)
            self.southeast.show(ax)
            self.southwest.show(ax)
    

class QuadTree:
    def __init__(self, points, x_min, x_max, y_min, y_max, max_pc_per_node=4) -> None:
        points_ = [Point(i, point) for i, point in enumerate(points)]
        # find the range for points
        self.max_pc_per_node = max_pc_per_node
        width = x_max - x_min
        height = y_max - y_min
        cnt_x = x_min + width / 2
        cnt_y = y_min + height / 2
        self.root = Node(cnt_x, cnt_y, width, height, max_pc_per_node)
        for point in points_:
            self.root.insert(point)
    def draw_all_grids(self, lines):
        def _search(node, lines):
            x_min = node.x - node.width / 2
            x_max = node.x + node.width / 2
            y_min = node.y - node.height / 2
            y_max = node.y + node.height / 2
            # 4 lines for each node
            lines.extend([(x_min, x_min, y_min, y_max),
                          (x_min, x_max, y_min, y_min),
                          (x_max, x_max, y_min, y_max),
                          (x_min, x_max, y_max, y_max)])
            if node.divided:
                _search(node.northeast, lines)
                _search(node.northwest, lines)
                _search(node.southeast, lines)
                _search(node.southwest, lines)
        _search(self.root, lines)
        return lines
        

