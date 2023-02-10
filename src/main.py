import os, PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Draw a white circle on a black background
# And save it as a png file
def draw_circle(radius, center=(128, 128), increasing=True, dir='data/circle', step=1):
    dir = f'{dir}_{step}'
    make_dir(dir)
    
    if increasing:
        range_params = range(0, radius, step)
        inc_text = 'inc'
    else:
        range_params = range(radius, 0, -step)
        inc_text = 'dec'
        
    for i in range_params:
        img = np.zeros((256, 256), dtype=np.uint8)
        circle_img = Image.fromarray(img)
        circle_draw = ImageDraw.Draw(circle_img)
        coords = (center[0]-i/2, center[1]-i/2, center[0]+i/2, center[1]+i/2)
        coords = tuple(map(int, coords))
        circle_draw.ellipse(coords, fill=255)
        if increasing:
            circle_img.save(f'{dir}/circle_{inc_text}_{str(i).zfill(3)}.png')
        else:
            circle_img.save(f'{dir}/circle_{inc_text}_{str(radius-i).zfill(3)}.png')
        
    
# Draw a white square on a black background
# and save it as a png file
def draw_square(side, center=(128, 128), increasing=True, step=1, dir='data/square'):
    dir = f'{dir}_{step}'
    make_dir(dir)
    
    if increasing:
        range_params = range(0, side, step)
        inc_text = 'inc'
    else:
        range_params = range(side, 0, -step)
        inc_text = 'dec'
        
    for i in range_params:
        img = np.zeros((256, 256), dtype=np.uint8)
        square_img = Image.fromarray(img)
        square_draw = ImageDraw.Draw(square_img)
        coords = (center[0]-i/2, center[1]-i/2, center[0]+i/2, center[1]+i/2)
        coords = tuple(map(int, coords))
        square_draw.rectangle(coords, fill=255)
        if increasing:
            square_img.save(f'{dir}/square_{inc_text}_{str(i).zfill(3)}.png')
        else:
            square_img.save(f'{dir}/square_{inc_text}_{str(side-i).zfill(3)}.png')

# Draw a rectangle with a 1:2 ratio
# White rectangle on a black background
# Save it as a png file
def draw_rectangle(side, center=(128, 128), increasing=True, step=1, dir='data/rectangle', x_ratio=1, y_ratio=2):
    dir = f'{dir}_{step}_x{x_ratio}_y{y_ratio}'
    make_dir(dir)
    
    if increasing:
        range_params = range(0, side, step)
        inc_text = 'inc'
    else:
        range_params = range(side, 0, -step)
        inc_text = 'dec'
        
    for i in range_params:
        img = np.zeros((256, 256), dtype=np.uint8)
        rectangle_img = Image.fromarray(img)
        rectangle_draw = ImageDraw.Draw(rectangle_img)
        coords = (center[0]-i/(2*x_ratio), center[1]-i/(2*y_ratio), center[0]+i/(2*x_ratio), center[1]+i/(2*y_ratio))
        coords = tuple(map(int, coords))
        rectangle_draw.rectangle(coords, fill=255)
        if increasing:
            rectangle_img.save(f'{dir}/rectangle_{inc_text}_{str(i).zfill(3)}.png')
        else:
            rectangle_img.save(f'{dir}/rectangle_{inc_text}_{str(side-i).zfill(3)}.png')

# Draw a ellipse in the center of the image on a black background
# Save it as a png file
def draw_ellipse(side, center=(128, 128), increasing=True, step=1, dir='data/ellipse', x_ratio=1, y_ratio=2):
    dir = 'data/ellipse'
    dir = f'{dir}_{step}_x{x_ratio}_y{y_ratio}'
    make_dir(dir)
    
    if increasing:
        range_params = range(0, side, step)
        inc_text = 'inc'
    else:
        range_params = range(side, 0, -step)
        inc_text = 'dec'
        
    for i in range_params:
        img = np.zeros((256, 256), dtype=np.uint8)
        ellipse_img = Image.fromarray(img)
        ellipse_draw = ImageDraw.Draw(ellipse_img)
        coords = (center[0]-i/(2*x_ratio), center[1]-i/(2*y_ratio), center[0]+i/(2*x_ratio), center[1]+i/(2*y_ratio))
        coords = tuple(map(int, coords))
        ellipse_draw.ellipse(coords, fill=255)
        if increasing:
            ellipse_img.save(f'{dir}/ellipse_{inc_text}_{str(i).zfill(3)}.png')
        else:
            ellipse_img.save(f'{dir}/ellipse_{inc_text}_{str(side-i).zfill(3)}.png')
            

# Draw a triangle in the center of the image on a black background
def draw_triangle(side, center=(128, 128), increasing=True, step=1, dir='data/triangle', up=True):
    dir = 'data/triangle'
    
    if up == True:
        up = 'up'
    else:
        up = 'down'
    dir = f'{dir}_{step}_{up}'
    
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    if increasing:
        range_params = range(0, side, step)
        inc_text = 'inc'
    else:
        range_params = range(side, 0, -step)
        inc_text = 'dec'
        
    for i in range_params:
        img = np.zeros((256, 256), dtype=np.uint8)
        triangle_img = Image.fromarray(img)
        triangle_draw = ImageDraw.Draw(triangle_img)
        if up == 'down':
            coords = (center[0], center[1]+i/2, center[0]+i/2, center[1]-i/2, center[0]-i/2, center[1]-i/2)
        else:
            coords = (center[0], center[1]-i/2, center[0]+i/2, center[1]+i/2, center[0]-i/2, center[1]+i/2)
        coords = tuple(map(int, coords))
        triangle_draw.polygon(coords, fill=255)
        if increasing:
            triangle_img.save(f'{dir}/triangle_{inc_text}_{str(i).zfill(3)}.png')
        else:
            triangle_img.save(f'{dir}/triangle_{inc_text}_{str(side-i).zfill(3)}.png')
            
            
# Draw a closed contour of random points on a black background
def draw_random_contour(side, center=(128, 128), increasing=True, step=1, dir='data/random_contour'):
    dir = 'data/random_contour'
    dir = f'{dir}_{step}'
    make_dir(dir)
    
    if increasing:
        range_params = range(0, side, step)
        inc_text = 'inc'
    else:
        range_params = range(side, 0, -step)
        inc_text = 'dec'
        
    


if __name__ == "__main__":
    for i in range(1, 2):
        # draw_circle(200, step=i, increasing=True)
        # draw_circle(200, step=i, increasing=False)
        
        # draw_square(200, step=i, increasing=True)
        # draw_square(200, step=i, increasing=False)

        # draw_rectangle(200, step=i, x_ratio=1, y_ratio=2, increasing=True)
        # draw_rectangle(200, step=i, x_ratio=1, y_ratio=2, increasing=False)
        # draw_rectangle(200, step=i, x_ratio=2, y_ratio=1, increasing=True)
        # draw_rectangle(200, step=i, x_ratio=2, y_ratio=1, increasing=False)

        # draw_ellipse(200, step=i, x_ratio=1, y_ratio=2, increasing=True)
        # draw_ellipse(200, step=i, x_ratio=1, y_ratio=2, increasing=False)
        # draw_ellipse(200, step=i, x_ratio=2, y_ratio=1, increasing=True)
        # draw_ellipse(200, step=i, x_ratio=2, y_ratio=1, increasing=False)
        
        # draw_triangle(200, step=i, up=False, increasing=True)
        # draw_triangle(200, step=i, up=False, increasing=False)
        # draw_triangle(200, step=i, up=True, increasing=True)
        # draw_triangle(200, step=i, up=True, increasing=False)
        