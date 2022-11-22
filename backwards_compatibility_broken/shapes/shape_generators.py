"""
Generates synthetic image data and labels for the "geometric shape" experiments.
"""
import cv2
import numpy as np
from numpy.random import default_rng
from shapes.shape_enums import *  # shape constants/enums
import random
import sys
random = default_rng()

IM_SZ = 512
BACKGROUND_LINE_THICKNESS = 10
NCOL = 3
NROW = 3
MAX_SHAPE_SZ = int(IM_SZ/max(NCOL, NROW))


def plan_rule_based_img(img_class):
    """
    Generate an image plan that obeys the rules of one of our 
    rule-based image classes. This does not guarantee that the plan returned
    will not obey any of the other rules! 
    """

    # plan image for 1 of four simple rule-based image classes
    img_plan = [[-1,-1,-1],
                [-1,-1,-1],
                [-1,-1,-1]]
    
    if img_class == 0:
        #class A--has exactly 1 parallelogram in every row
        for r in range(NROW):
            c = random.integers(0, NCOL)
            img_plan[r][c] = np.random.choice(PARALLELOGRAMS)
        fill_shapes = [NO_SHAPE, NO_SHAPE, NO_SHAPE, NO_SHAPE, CIRCLE, OVAL, 
                         TRIANGLE, PENTAGON, HEXAGON, OCTAGON]
    elif img_class == 1:
        # same number of ellipse and non-ellipse 
        num_ellipse = np.random.choice([1,2,3])
        coords = []
        for row in range(NROW):
            for col in range(NCOL):
                coords.append((row, col))
        random.shuffle(coords)
        for ei in range(num_ellipse):
            coord = coords.pop()
            img_plan[coord[0]][coord[1]] = np.random.choice(ELLIPSES)
        for bi in range(NCOL*NROW - 2*num_ellipse):
            coord = coords.pop()
            img_plan[coord[0]][coord[1]] = NO_SHAPE
            fill_shapes = [SQUARE, RECT, SKEW_RECT, 
                         PENTAGON, HEXAGON, OCTAGON]    
    elif img_class == 2:
        # complete row/column of polygons
        if np.random.random() < 0.5:
            col = random.integers(0, NCOL)
            coords = [(i,col) for i in range(NROW)]
        else:
            row = random.integers(0, NROW)
            coords = [(row, i) for i in range(NCOL)]
        for coord in coords:
            img_plan[coord[0]][coord[1]] = np.random.choice(POLYGONS)
        fill_shapes = [NO_SHAPE, NO_SHAPE, NO_SHAPE, NO_SHAPE, NO_SHAPE, SQUARE, RECT, CIRCLE, OVAL, 
                        SKEW_RECT, TRIANGLE, PENTAGON, HEXAGON, OCTAGON]        
    elif img_class == 3:
        # class D-- has exactly 1 triangle in the center
        center = (NROW-1)//2, (NCOL-1)//2
        img_plan[center[0]][center[1]] = TRIANGLE
        fill_shapes = [NO_SHAPE, NO_SHAPE, NO_SHAPE, NO_SHAPE, SQUARE, RECT, CIRCLE, OVAL, SKEW_RECT, 
                         PENTAGON, HEXAGON, OCTAGON]        
    else:
        raise ValueError
        
    # fill in remaining undetermined spaces?
    for r in range(NROW):
        for c in range(NCOL):
            if img_plan[r][c] == -1:
                img_plan[r][c] = np.random.choice(fill_shapes)
        
    return img_plan
    
    
def obeys_class(img_plan, img_class):
    """
    Returns whether an image plan obeys the rules for a specific image class. 
    Note that a single call to this does not tell you whether the specified image plan might
    obey rules for the other image classes as well! You will have to make multiple calls to
    check that.
    """

    if img_class == 0:
        # 1 parallelogram in every row 
        for row in img_plan:
            pcount = 0
            for shape in row:
                pcount += int(shape in PARALLELOGRAMS)
            if pcount != 1:
                return False
        return True
    elif img_class == 1:
        # same number of ellipse and non-ellipse
        nellipse = 0
        nnonellipse = 0
        for row in img_plan:
            for shape in row:
                nellipse += int(shape in ELLIPSES)
                nnonellipse += int(shape != NO_SHAPE and shape not in ELLIPSES)
        return nellipse == nnonellipse
    elif img_class == 2:
        # complete row/column of polygons
        for row in img_plan:
            if all([shape in POLYGONS for shape in row]):
                return True 
        for col_i in range(NCOL):
            if all([img_plan[i][col_i] in POLYGONS for i in range(NROW)]):
                return True
        return False
    elif img_class == 3:
        # exactly 1 traingle in center
        center = (NROW-1)//2, (NCOL-1)//2
        if img_plan[center[0]][center[1]] != TRIANGLE:
            return False
        for r in range(NROW):
            for c in range(NCOL):
                if (r, c) != center and img_plan[r][c] == TRIANGLE:
                    return False
        return True
    else:
        raise ValueError


def draw_image(img_plan, color_func):
    assert np.array(img_plan).shape == (NROW, NCOL)

    img = 255 * np.ones(shape=[IM_SZ, IM_SZ, 3], dtype=np.uint8)
    sem_seg = np.zeros(shape=[IM_SZ, IM_SZ, 1], dtype=np.uint8) + NO_SHAPE
    
    # draw backgrounds to make confusing edge gradients 
    for x in range(0, IM_SZ, BACKGROUND_LINE_THICKNESS*2):
        cv2.line(img, (x, 0), (x, IM_SZ), (0, 0, 0), BACKGROUND_LINE_THICKNESS)
    
    for row in range(NROW):
        for col in range(NCOL):
            r = int(MAX_SHAPE_SZ*(row+0.5))
            c = int(MAX_SHAPE_SZ*(col+0.5))
            shape_type = img_plan[row][col]
            color = color_func(shape_type) if shape_type != NO_SHAPE else [0,0,0]
            
            if shape_type == NO_SHAPE:
                continue
            elif shape_type == SQUARE:
                draw_square(img, sem_seg, r, c, color)
            elif shape_type == RECT:
                draw_rect(img, sem_seg, r, c, color)
            elif shape_type == CIRCLE:
                draw_circle(img, sem_seg, r, c, color)
            elif shape_type == OVAL:
                draw_oval(img, sem_seg, r, c, color)
            elif shape_type == SKEW_RECT:
                draw_skew_rect(img, sem_seg, r, c, color)
            elif shape_type == TRIANGLE:
                draw_triangle(img, sem_seg, r, c, color)
            elif shape_type == PENTAGON:
                draw_pentagon(img, sem_seg, r, c, color)
            elif shape_type == HEXAGON:
                draw_hexagon(img, sem_seg, r, c, color)
            elif shape_type == OCTAGON:
                draw_octagon(img, sem_seg, r, c, color)
            else:
                raise ValueError
        
    return img, sem_seg
        
        
def draw_square(img, sem_seg, r, c, color):
    square_rad = int(random.uniform(0.1, 0.9)*MAX_SHAPE_SZ/2)
    cv2.rectangle(img, pt1=(c-square_rad, r-square_rad), pt2=(c+square_rad, r+square_rad), color=color, thickness=-1)
    cv2.rectangle(sem_seg, pt1=(c-square_rad, r-square_rad), pt2=(c+square_rad, r+square_rad), color=SQUARE, thickness=-1)


def draw_rect(img, sem_seg, r, c, color):
    half_width = int(random.uniform(0.1, 0.9)*MAX_SHAPE_SZ/2)
    half_height = half_width
    while (0.8 < half_width/half_height < 1.2):
        half_height = int(random.uniform(0.1, 0.9)*MAX_SHAPE_SZ/2)
    cv2.rectangle(img, pt1=(c-half_width, r-half_height), pt2=(c+half_width, r+half_height), color=color, thickness=-1)
    cv2.rectangle(sem_seg, pt1=(c-half_width, r-half_height), pt2=(c+half_width, r+half_height), color=RECT, thickness=-1)

    
def draw_circle(img, sem_seg, r, c, color):
    radius = int(random.uniform(0.1, 0.9)*MAX_SHAPE_SZ/2)
    cv2.circle(img, (c, r), radius, color=color, thickness=-1)
    cv2.circle(sem_seg, (c, r), radius, color=CIRCLE, thickness=-1)
    
    
def draw_oval(img, sem_seg, r, c, color):
    minor_axis = int(random.uniform(0.1, 0.7)*MAX_SHAPE_SZ/2)
    major_axis = int(random.uniform(minor_axis*1.3, 0.9*MAX_SHAPE_SZ/2))
    rot = random.integers(0,360)
    cv2.ellipse(img, (c,r), (minor_axis, major_axis), rot, 0, 360, color, thickness=-1)
    cv2.ellipse(sem_seg, (c,r), (minor_axis, major_axis), rot, 0, 360, OVAL, thickness=-1)
    
    
def draw_skew_rect(img, sem_seg, r, c, color):
    half_width = int(random.uniform(0.2, 0.7)*MAX_SHAPE_SZ/2)
    half_height = int(random.uniform(0.2, 0.7)*MAX_SHAPE_SZ/2)
    x_skew = 0
    y_skew = 0
    while (abs(x_skew) < 5 and abs(y_skew) < 5):
        x_bound = min(MAX_SHAPE_SZ - 2*half_width, half_width/2)
        y_bound = min(MAX_SHAPE_SZ - 2*half_height, half_height/2)
        x_skew = int(random.uniform(-x_bound, x_bound))
        y_skew = int(random.uniform(-y_bound, y_bound))
    pts = np.array([[c - half_width + x_skew, r - half_height + y_skew],
                    [c + half_width + x_skew, r - half_height - y_skew],
                    [c + half_width - x_skew, r + half_height - y_skew],
                    [c - half_width - x_skew, r + half_height + y_skew]])
    cv2.fillConvexPoly(img, pts, color=color)
    cv2.fillConvexPoly(sem_seg, pts, color=SKEW_RECT)
    
    
def draw_triangle(img, sem_seg, r, c, color):
    # we can pick any 3 points! --as long as they aren't too close to colinear
    accept = False
    while not accept:
        pts = np.array([[np.random.uniform(c-MAX_SHAPE_SZ/2,c+MAX_SHAPE_SZ/2),np.random.uniform(r-MAX_SHAPE_SZ/2,r+MAX_SHAPE_SZ/2)],
                        [np.random.uniform(c-MAX_SHAPE_SZ/2,c+MAX_SHAPE_SZ/2),np.random.uniform(r-MAX_SHAPE_SZ/2,r+MAX_SHAPE_SZ/2)],
                        [np.random.uniform(c-MAX_SHAPE_SZ/2,c+MAX_SHAPE_SZ/2),np.random.uniform(r-MAX_SHAPE_SZ/2,r+MAX_SHAPE_SZ/2)]])
        pts = pts.astype(np.int32)
        #accept = abs((pts[2,1] - pts[1,1])*(pts[1,0] - pts[0,0]) - (pts[1,1] - pts[0,1])*(pts[2,0] - pts[1,0])) > 0.25
        area = 0.5 * (pts[0,0] * (pts[1,1] - pts[2,1]) + pts[1,0] * (pts[2,1] - pts[0,1]) + pts[2,0] * (pts[0,1] - pts[1,1]))
        longest_side = max(np.linalg.norm(pts[0,:] - pts[1,:]), np.linalg.norm(pts[1,:] - pts[2,:]), np.linalg.norm(pts[0,:] - pts[2,:]))
        accept = 0.5 * (pts[0,0] * (pts[1,1] - pts[2,1]) + pts[1,0] * (pts[2,1] - pts[0,1]) + pts[2,0] * (pts[0,1] - pts[1,1])) > longest_side*longest_side*0.15
    cv2.fillConvexPoly(img, pts, color=color)
    cv2.fillConvexPoly(sem_seg, pts, color=TRIANGLE)
    
    
def draw_pentagon(img, sem_seg, r, c, color):
    pts = regular_polygon_pts(r, c, 5)
    cv2.fillConvexPoly(img, pts, color=color)
    cv2.fillConvexPoly(sem_seg, pts, color=PENTAGON)
    
    
def draw_hexagon(img, sem_seg, r, c, color):
    pts = regular_polygon_pts(r, c, 6)
    cv2.fillConvexPoly(img, pts, color=color)
    cv2.fillConvexPoly(sem_seg, pts, color=HEXAGON)
    
    
def draw_octagon(img, sem_seg, r, c, color):
    pts = regular_polygon_pts(r, c, 8)
    cv2.fillConvexPoly(img, pts, color=color)
    cv2.fillConvexPoly(sem_seg, pts, color=OCTAGON)


def regular_polygon_pts(r, c, nvertices):
    assert nvertices > 2
    radius = int(random.uniform(0.1, 0.9)*MAX_SHAPE_SZ/2)
    # evenly divide angle
    pts = []
    for vertex in range(nvertices):
        angle = vertex*np.pi*2/nvertices
        pts.append([int(c + radius*np.sin(angle)), int(r + radius*np.cos(angle))])
    return np.array(pts)
    
    
def gen_color():
    accept = False
    while not accept:
        color = np.array([np.random.choice([0,128,255]),
                np.random.choice([0,128,255]),
                np.random.choice([0,128,255])])
        if np.array_equal(color, [0,0,0]):
            accept = False
        elif np.array_equal(color, [255, 255, 255]):
            accept = False 
        else:
            accept = True
    return color.tolist()
    

def gen_color_weighted(seed):
    assert 1 <= seed <= 9
    
    preset_colors =[[255,0,0],
                    [0,255,0],
                    [0,0,255],
                    [255,255,0],
                    [0,255,255],
                    [255,0,255],
                    [0,128,255],
                    [255,0,128],
                    [128,255,0]]
                    
    if np.random.random() < 0.5:
        return preset_colors[seed-1]
    else:
        return gen_color()
                    
    
def drawing_test():
    for i in range(25):
        img_plan = [[SQUARE,RECT,CIRCLE],
                    [OVAL, SKEW_RECT, TRIANGLE],
                    [PENTAGON, HEXAGON, OCTAGON]]
        img, sem_seg = draw_image(img_plan)
        cv2.imwrite(f"drawing_test_{i}.png", img)


def gen_trainset(setsize: int, savedir):
    for im_num in range(setsize):
        img_class = im_num % NUM_IM_CLASSES
        accept_plan = False
        while not accept_plan:
            img_plan = plan_rule_based_img(img_class)
            accept_plan = True
            for c in range(NUM_IM_CLASSES):
                if c == img_class:
                    assert obeys_class(img_plan, c)
                elif obeys_class(img_plan, c):
                    accept_plan = False
        img, sem_seg = draw_image(img_plan, lambda shape: gen_color_weighted(shape))
        cv2.imwrite(savedir + f"/{im_num}.png", img)
        cv2.imwrite(savedir + f"/{im_num}_seg.png", sem_seg)


def gen_valset(setsize: int, savedir):
    for im_num in range(setsize):
        img_class = im_num % NUM_IM_CLASSES
        accept_plan = False
        while not accept_plan:
            img_plan = plan_rule_based_img(img_class)
            accept_plan = True
            for c in range(NUM_IM_CLASSES):
                if c == img_class:
                    assert obeys_class(img_plan, c)
                elif obeys_class(img_plan, c):
                    accept_plan = False
        img, sem_seg = draw_image(img_plan, lambda shape: gen_color_weighted(shape+2 if shape+2 <= NUM_SHAPES else shape-7))
        cv2.imwrite(savedir + f"/{im_num}.png", img)
        cv2.imwrite(savedir + f"/{im_num}_seg.png", sem_seg)

        
def gen_testset(setsize: int, savedir):
    for im_num in range(setsize):
        img_class = im_num % NUM_IM_CLASSES
        accept_plan = False
        while not accept_plan:
            img_plan = plan_rule_based_img(img_class)
            accept_plan = True
            for c in range(NUM_IM_CLASSES):
                if c == img_class:
                    assert obeys_class(img_plan, c)
                elif obeys_class(img_plan, c):
                    accept_plan = False
        img, sem_seg = draw_image(img_plan, lambda shape: gen_color_weighted(shape+1 if shape+1 <= NUM_SHAPES else shape-8))
        cv2.imwrite(savedir + f"/{im_num}.png", img)
        cv2.imwrite(savedir + f"/{im_num}_seg.png", sem_seg)


    
if __name__ == '__main__':
    gen_trainset(int(sys.argv[1]), sys.argv[2])