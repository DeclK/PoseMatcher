import cv2
import numpy as np
from skimage import draw, io
from PIL import Image, ImageDraw, ImageFont
from easydict import EasyDict
from mmengine.visualization import Visualizer
from typing import Union
from utils import get_skeleton

# build pallatte
def pallete():
    colors = [(183, 211, 50), (158, 188, 25), (132, 167, 41)]
    pallete = np.zeros((256, 3), dtype=np.uint8)
    pallete[:len(colors)] = colors
    return pallete


class FastVisualizer:
    """ Use skimage to draw, which is much faster than matplotlib, and 
    more beatiful than opencv.😎
    """
    # TODO: set color pallete
    def __init__(self, image=None) -> None:
        self.set_image(image)
        self.colors = self.get_pallete()
        self.skeleton = get_skeleton()

    def set_image(self, image: Union[str, np.ndarray]):
        if isinstance(image, str):
            self.image = cv2.imread(image)
        elif isinstance(image, np.ndarray) or image is None:
            self.image = image
        else:
            raise TypeError(f"Type {type(image)} is not supported")

    def get_image(self):
        return self.image


    def draw_box(self, box_coord, color=0, alpha=1.0):
        """ Draw a box on the image
        Args:
            box_coord: a list of [xmin, ymin, xmax, ymax]
            alpha: the alpha of the box
            color: the edge color of the box
        """
        xmin, ymin, xmax, ymax = box_coord
        rr, cc = draw.rectangle_perimeter((ymin, xmin), (ymax, xmax))
        draw.set_color(self.image, (rr, cc), color, alpha=alpha)
        return self

    def draw_rectangle(self, box_coord, color=0, alpha=1.0):
        xmin, ymin, xmax, ymax = box_coord
        rr, cc = draw.rectangle((ymin, xmin), (ymax, xmax))
        draw.set_color(self.image, (rr, cc), color, alpha=alpha)
        return self

    def draw_point(self, point_coord, color=0, alpha=1.0):
        rr, cc = draw.disk(point_coord, radius=5)
        draw.set_color(self.image, (rr, cc), color, alpha=alpha)
        return self

    def draw_line(self, start_point, end_point, color=0, alpha=1.0):
        rr, cc = draw.line(start_point[0], start_point[1],
                           end_point[0], end_point[1])
        draw.set_color(self.image, (rr, cc), color, alpha)
        return self
        
    def draw_boxes(self, boxes, alpha=0.5, color=0):
        for box in boxes:
            self.draw_box(box, alpha, color)
        return self
    
    def draw_rectangles(self, boxes, alpha=0.5, color=0):
        for box in boxes:
            self.draw_rectangle(box, alpha, color)
        return self
    
    def draw_points(self, points, alpha=0.5, color=0):
        for point in points:
            self.draw_point(point, alpha, color)
        return self

    def draw_text(self, text, position,
                  font_path='assets/SmileySans/SmileySans-Oblique.ttf',
                  font_size=20,
                  text_color=(255, 255, 255)):
        """ Position is the left top corner of the text
        """
        # Convert the NumPy array to a PIL image
        pil_image = Image.fromarray(np.uint8(self.image))
        # Load the font (default is Arial)
        font = ImageFont.truetype(font_path, font_size)
        # Create a drawing object
        draw = ImageDraw.Draw(pil_image)
        # Add the text to the image
        draw.text(position, text, font=font, fill=text_color)
        # Convert the PIL image back to a NumPy array
        result = np.array(pil_image)

        self.image = result
        return self
    
    def draw_skeleton(self, keypoints, scores):
        """ Draw skeleton on the image, and give different color according
        to similarity scores.
        """
        for key, links in self.skeleton.items():
            if key == 'head':
                pass
            elif key == 'left_arm':
                pass
            elif key == 'right_arm':
                pass
            elif key == 'left_leg':
                pass
            elif key == 'right_leg':
                pass
            elif key == 'body':
                pass
            else:
                raise KeyError(f"Key {key} is not supported")
            links = np.array(links)
            start_points = keypoints[links[0]]
            end_points = keypoints[links[1]]
            for s, e in zip(start_points, end_points):
                self.draw_line(s, e, color=self.colors[key], alpha=0.8)
            
            

        return self


    def draw_score_bar(self, score, factor=50, bar_ratio=7):
        """ Draw a score bar on the left top of the image.
        factor: the value of image longer edge divided by the bar height
        bar_ratio: the ratio of bar width to bar height
        """
        # calculate bar's height and width
        long_edge = np.max(self.image.shape[:2])
        short_edge = np.min(self.image.shape[:2])
        bar_h = long_edge // factor
        bar_w = bar_h * bar_ratio
        if bar_w * 3 > short_edge:
            # when the image width is not enough
            bar_w = short_edge // 4
            bar_h = bar_w // bar_ratio
        cube_size = bar_h
        # bar's base position
        bar_start_point = (2*bar_h, 2*bar_h)
        # draw bar horizontally, and record the position of each word
        word_positions = []
        box_coords = []
        colors = [self.colors.bad, self.colors.good, self.colors.vamos]
        for i, color in enumerate(colors):
            x0, y0 = bar_start_point[0] + i*bar_w,  bar_start_point[1]
            x1, y1 = x0 + bar_w - 1,  y0 + bar_h
            box_coord = np.array((x0, y0, x1, y1))
            self.draw_rectangle(box_coord, color=color)

            box_coords.append(box_coord)
            word_positions.append(np.array((x0, y1 + bar_h // 2)))
        # calculate cube position according to score
        lvl, lvl_ratio, lvl_name = self.score_level(score)
        # the first level start point is the first bar
        cube_lvl_start_x0 = [box_coord[0] - cube_size // 2 if i != 0 
                             else box_coord[0] 
                             for i, box_coord in enumerate(box_coords)]
        # process the last level, I want the cube stays in the bar
        level_length = bar_w if lvl == 1 else bar_w - cube_size // 2
        cube_x0 = cube_lvl_start_x0[lvl] + lvl_ratio * level_length
        cube_y0 = bar_start_point[1] - bar_h // 2 - cube_size
        cube_x1 = cube_x0 + cube_size
        cube_y1 = cube_y0 + cube_size
        # draw cube
        self.draw_rectangle((cube_x0, cube_y0, cube_x1, cube_y1),
                             color=self.colors.cube)
        # enlarge the box, to emphasize the level
        enlarged_box = box_coords[lvl].copy()
        enlarged_box[:2] = enlarged_box[:2] - bar_h // 8
        enlarged_box[2:] = enlarged_box[2:] + bar_h // 8
        self.draw_rectangle(enlarged_box, color=self.colors[lvl_name])

        # draw text
        if lvl_name == 'vamos':
            lvl_name = 'vamos!!'    # exciting!
        self.draw_text(lvl_name.capitalize(),
                       word_positions[lvl],
                       font_size=bar_h * 2,
                       text_color=tuple(colors[lvl].tolist()))

        return self

    def score_level(self, score):
        """ Return the level according to score.
        bad: 0.00 - 0.50; good: 0.50 - 0.85; vamos: 0.85 - 1.00
        """
        if score < 0.5: return 0, score / 0.5, 'bad'
        elif score < 0.85: return 1, (score - 0.5) / 0.35, 'good'
        else: return 2, (score - 0.85) / 0.15, 'vamos'
    
    def get_pallete(self):
        PALLETE = EasyDict()
        
        PALLETE.bad = np.array([253, 138, 138])
        PALLETE.good = np.array([168, 209, 209])
        PALLETE.vamos = np.array([241, 247, 181])
        PALLETE.cube = np.array([158, 161, 212])

        PALLETE.left_arm = np.array([218, 119, 242])
        PALLETE.right_arm = np.array([151, 117, 250])
        PALLETE.left_leg = np.array([255, 212, 59])
        PALLETE.right_leg = np.array([255, 169, 77])

        PALLETE.head = np.array([255, 255, 255])
        PALLETE.body = np.array([255, 255, 255])

        return PALLETE
    
if __name__ == '__main__':
    vis =  FastVisualizer()

    image = '/github/Tennis.ai/assets/test.png'
    import matplotlib.pyplot as plt
    import mmcv
    image = mmcv.imread(image, channel_order='rgb')
    # image = io.imread(image)
    vis.set_image(image)
    vis.draw_score_bar(0.20)
    plt.imshow(vis.image)
    plt.show()