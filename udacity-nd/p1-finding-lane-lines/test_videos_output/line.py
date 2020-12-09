import numpy
import cv2
# line object with helper slope 
class Line:
    """
    line connecting (x1, y1) and (x2, y2)
    # https://www.mathsisfun.com/algebra/line-equation-2points.html
    """
    def __init__(self, x1, y1, x2, y2):

        self.x1 = np.float32(x1)
        self.y1 = np.float32(y1)
        self.x2 = np.float32(x2)
        self.y2 = np.float32(y2)

        self.slope = self._slope()
        self.bias = self._bias()

    def _slope(self):
        # line slope https://www.mathsisfun.com/geometry/slope.html
        return (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)

    def _bias(self):
        # from slope equation => bias => y = mx + b => b = (y - mx)
        return self.y1 - self.slope * self.x1

    def _center(self):
      self.center_x = (self.x1 + self.x2 ) / 2
      self.center_y = (self.y1 + self.y2 ) / 2

    def get_coords(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def set_coords(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self, img, color=[0, 255, 0], thickness=2):
        cv2.line(img, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), color, thickness)