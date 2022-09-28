"""
image class indices for semantic segmentation ground truth and image generation in the "geometric shape experiments".
"""
NO_SHAPE = 0
SQUARE = 1
RECT = 2
CIRCLE = 3
OVAL = 4
SKEW_RECT = 5
TRIANGLE = 6
PENTAGON = 7
HEXAGON = 8
OCTAGON = 9
NUM_SHAPES = 9

NUM_IM_CLASSES = 4

# mapping to 'high-level' shape categories
ELLIPSES = [CIRCLE, OVAL]
PARALLELOGRAMS = [SQUARE, RECT, SKEW_RECT]
POLYGONS = [SQUARE, RECT, SKEW_RECT, TRIANGLE, PENTAGON, HEXAGON, OCTAGON]