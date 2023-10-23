import cv2
import numpy as np
from skimage import draw, io
from PIL import Image, ImageDraw, ImageFont
from easydict import EasyDict
from typing import Union
from .utils import get_skeleton, Timer

class FastVisualizer:
    """ Use skimage to draw, which is much faster than matplotlib, and 
    more beatiful than opencv.😎
    """
    # TODO: modify color input parameter
    def __init__(self, image=None) -> None:
        self.set_image(image)
        self.colors = self.get_pallete()
        self.skeleton = get_skeleton()
        self.lvl_tresh = self.set_level([0.3, 0.6, 0.8])

    def set_image(self, image: Union[str, np.ndarray]):
        if isinstance(image, str):
            self.image = cv2.imread(image)
        elif isinstance(image, np.ndarray) or image is None:
            self.image = image
        else:
            raise TypeError(f"Type {type(image)} is not supported")

    def get_image(self):
        return self.image

    def draw_box(self, box_coord, color=(25, 113, 194), alpha=1.0):
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

    def draw_rectangle(self, box_coord, color=(25, 113, 194), alpha=1.0):
        xmin, ymin, xmax, ymax = box_coord
        rr, cc = draw.rectangle((ymin, xmin), (ymax, xmax))
        draw.set_color(self.image, (rr, cc), color, alpha=alpha)
        return self

    def draw_point(self, point_coord, radius=5, color=(25, 113, 194), alpha=1.0):
        """ Coord in (x, y) format, but will be converted to (y, x)
        """
        x, y = point_coord
        rr, cc = draw.disk((y, x), radius=radius)
        draw.set_color(self.image, (rr, cc), color, alpha=alpha)
        return self

    def draw_line(self, start_point, end_point, color=(25, 113, 194), alpha=1.0):
        """ Not used, because I can't produce smooth line.
        """
        cv2.line(self.image, start_point, end_point, color.tolist(), 2,
                 cv2.LINE_AA)
        return self

    def draw_line_aa(self, start_point, end_point, color=(25, 113, 194), alpha=1.0):
        """ Not used, because I can't produce smooth line.
        """
        x1, y1 = start_point
        x2, y2 = end_point
        rr, cc, val = draw.line_aa(y1, x1, y2, x2)
        draw.set_color(self.image, (rr, cc), color, alpha=alpha)
        return self

    def draw_thick_line(self, start_point, end_point, thickness=1, color=(25, 113, 194), alpha=1.0):
        """ Not used, because I can't produce smooth line.
        """
        x1, y1 = start_point
        x2, y2 = end_point
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        cos, sin = dx / length, dy / length

        half_t = thickness / 2.0
        # Calculate the polygon vertices
        vertices_x = [x1 - half_t * sin, x1 + half_t * sin,
                        x2 + half_t * sin, x2 - half_t * sin]
        vertices_y = [y1 + half_t * cos, y1 - half_t * cos,
                        y2 - half_t * cos, y2 + half_t * cos]
        rr, cc = draw.polygon(vertices_y, vertices_x)
        draw.set_color(self.image, (rr, cc), color, alpha)

        return self

    def draw_text(self, text, position,
                  font_path='assets/SmileySans-Oblique.ttf',
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

    def xyhw_to_xyxy(self, box):
        hw = box[2:]
        x1y1 = box[:2] - hw / 2
        x2y2 = box[:2] + hw / 2
        return np.concatenate([x1y1, x2y2]).astype(np.int32)
        
    def draw_line_in_discrete_style(self, start_point, end_point, size=2, sample_points=3, 
                                    color=(25, 113, 194), alpha=1.0):
        """ When drawing continous line, it is super fuzzy, and I can't handle them
        very well even tried OpneCV & PIL all kinds of ways. This is a workaround.
        The discrete line will be represented with few sampled cubes along the line,
        and it is exclusive with start & end points.
        """
        # sample points
        points = np.linspace(start_point, end_point, sample_points + 2)[1:-1]
        for p in points:
            rectangle_xyhw = np.array((p[0], p[1], size, size))
            rectangle_xyxy = self.xyhw_to_xyxy(rectangle_xyhw)
            self.draw_rectangle(rectangle_xyxy, color, alpha)
        return self
    
    def draw_human_keypoints(self, keypoints, scores=None, factor=20, draw_skeleton=False):
        """ Draw skeleton on the image, and give different color according
        to similarity scores.
        """
        # get max length of skeleton
        max_x, max_y = np.max(keypoints, axis=0)
        min_x, min_y = np.min(keypoints, axis=0)
        max_length = max(max_x - min_x, max_y - min_y)
        if max_length < 1: return self
        cube_size = max_length // factor
        line_cube_size = cube_size // 2
        # draw skeleton in discrete style
        if draw_skeleton:
            for key, links in self.skeleton.items():
                links = np.array(links)
                start_points = keypoints[links[:, 0]]
                end_points = keypoints[links[:, 1]]
                for s, e in zip(start_points, end_points):
                    self.draw_line_in_discrete_style(s, e, line_cube_size,
                    color=self.colors[key], alpha=0.9)
        # draw points
        if scores is None:  # use vamos color
            lvl_names = ['vamos'] * len(keypoints)
        else: lvl_names = self.score_level_names(scores)

        for idx, (point, lvl_name) in enumerate(zip(keypoints, lvl_names)):
            if idx in set((0, 1, 2, 3, 4)): 
                continue # do not draw head
            rectangle_xyhw = np.array((point[0], point[1], cube_size, cube_size))
            rectangle_xyxy = self.xyhw_to_xyxy(rectangle_xyhw)
            self.draw_rectangle(rectangle_xyxy, 
                                color=self.colors[lvl_name], 
                                alpha=0.8)
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
            box_coord = np.array((x0, y0, x1, y1), dtype=np.int32)
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
        cube_x0 , cube_y0 , cube_x1 , cube_y1 = map(int, (cube_x0 , cube_y0 , cube_x1 , cube_y1))
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
    
    
    
    def get_sim(self , score):
        t = self.lvl_tresh
        level = 0 if  score <= t[1] else (1 if score <= t[2] else 2)
        within_ratio = 0.0
        if score < t[1]: # t[0] might bigger than 0
            within_ratio = (score - t[0]) / (t[1] - t[0])
            within_ratio = np.clip(within_ratio, a_min=0, a_max=1)
        elif score < t[2]: 
            within_ratio = (score - t[1]) / (t[2] - t[1])
        else: 
            within_ratio = (score - t[2]) / (1 - t[2])
        return level , within_ratio , ((score - t[0])/(1 - t[0]) if score > t[0] else 0.0)
    
    def get_name(self , level):
        look_list = ["bad", "good", "vamos"]
        return look_list[level]
            
        
    
    
    def draw_sim_bar(self, score, factor=50, bar_ratio=7):
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
            box_coord = np.array((x0, y0, x1, y1), dtype=np.int32)
            self.draw_rectangle(box_coord, color=color)

            box_coords.append(box_coord)
            word_positions.append(np.array((x0, y1 + bar_h // 2)))
        # calculate cube position according to score
        lvl , within_ratio , sim_ratio = self.get_sim(score)
        #lvl, lvl_ratio, lvl_name = self.score_level(score)
        # the first level start point is the first bar
        cube_lvl_start_x0 = [box_coord[0] - cube_size // 2 if i != 0 
                             else box_coord[0] 
                             for i, box_coord in enumerate(box_coords)]
        # process the last level, I want the cube stays in the bar
        level_length = bar_w if lvl == 1 else bar_w - cube_size // 2
        cube_x0 = cube_lvl_start_x0[lvl] + within_ratio * level_length
        cube_y0 = bar_start_point[1] - bar_h // 2 - cube_size
        cube_x1 = cube_x0 + cube_size
        cube_y1 = cube_y0 + cube_size
        cube_x0 , cube_y0 , cube_x1 , cube_y1 = map(int, (cube_x0 , cube_y0 , cube_x1 , cube_y1))
        # draw cube
        self.draw_rectangle((cube_x0, cube_y0, cube_x1, cube_y1),
                             color=self.colors.cube)
        # enlarge the box, to emphasize the level
        enlarged_box = box_coords[lvl].copy()
        enlarged_box[:2] = enlarged_box[:2] - bar_h // 8
        enlarged_box[2:] = enlarged_box[2:] + bar_h // 8
        self.draw_rectangle(enlarged_box, color=self.colors[self.get_name(lvl)])

        # draw text
        # if lvl_name == 'vamos':
        #     lvl_name = 'vamos!!'    # exciting!
        lvl_name = "Sim = " + str(int(sim_ratio*100)) + "%"
        self.draw_text(lvl_name.capitalize(),
                       word_positions[lvl],
                       font_size=bar_h * 2,
                       text_color=tuple(colors[lvl].tolist()))

        return self

    def draw_non_transparent_area(self, box_coord, alpha=0.2, extend_ratio=0.1):
        """ Make image outside the box transparent using alpha blend
        """
        x1, y1, x2, y2 = box_coord.astype(np.int32)
        # enlarge the box for 10%
        max_len = max((x2 - x1), (y2 - y1))
        extend_len = int(max_len * extend_ratio)
        x1, y1 = x1 - extend_len, y1 - extend_len 
        x2, y2 = x2 + extend_len, y2 + extend_len
        # clip the box
        h, w = self.image.shape[:2]
        x1, y1, x2, y2 = np.clip((x1,y1,x2,y2), a_min=0, 
                                                a_max=(w,h,w,h))
        # Create a white background color
        bg_color = np.ones_like(self.image) * 255
        # Copy the box region from the image
        bg_color[y1:y2, x1:x2] = self.image[y1:y2, x1:x2]
        # Alpha blend inplace
        self.image[:] = self.image * alpha + bg_color * (1 - alpha)
        return self

    def draw_logo(self, logo='assets/logo.png', factor=30, shift=20):
        """ Draw logo on the right bottom of the image.
        """
        H, W = self.image.shape[:2]
        # load logo
        logo_img = Image.open(logo)
        # scale logo
        logo_h = self.image.shape[0] // factor
        scale_size = logo_h / logo_img.size[1]
        logo_w = int(logo_img.size[0] * scale_size)
        logo_img = logo_img.resize((logo_w, logo_h))
        # convert to RGBA
        image = Image.fromarray(self.image).convert("RGBA")
        # alpha blend
        image.alpha_composite(logo_img, (W - logo_w - shift,
                                         H - logo_h - shift))
        self.image = np.array(image.convert("RGB"))
        return self

    def score_level(self, score):
        """ Return the level according to level thresh.
        """
        t = self.lvl_tresh
        if score < t[1]: # t[0] might bigger than 0
            ratio = (score - t[0]) / (t[1] - t[0])
            ratio = np.clip(ratio, a_min=0, a_max=1)
            return 0, ratio, 'bad'
        elif score < t[2]: 
            ratio = (score - t[1]) / (t[2] - t[1])
            return 1, ratio, 'good'
        else: 
            ratio = (score - t[2]) / (1 - t[2])
            return 2, ratio, 'vamos'

    def score_level_names(self, scores):
        """ Get multiple score level, return numpy array.
        np.vectorize does not speed up loop, but it is convenient.
        """
        t = self.lvl_tresh
        func_lvl_name = lambda x: 'bad' if x < t[1] else 'good' \
                                        if x < t[2] else 'vamos'
        lvl_names = np.vectorize(func_lvl_name)(scores)
        return lvl_names
    
    def set_level(self, thresh):
        """ Set level thresh for bad, good, vamos.
        """
        from collections import namedtuple
        Level = namedtuple('Level', ['zero', 'good', 'vamos'])
        return Level(thresh[0], thresh[1], thresh[2])

    def get_pallete(self):
        PALLETE = EasyDict()
        
        # light set
        # PALLETE.bad = np.array([253, 138, 138])
        # PALLETE.good = np.array([168, 209, 209])
        # PALLETE.vamos = np.array([241, 247, 181])
        # PALLETE.cube = np.array([158, 161, 212])

        # dark set, set 80% brightness
        PALLETE.bad = np.array([204, 111, 111])
        PALLETE.good = np.array([143, 179, 179])
        PALLETE.vamos = np.array([196, 204, 124])
        PALLETE.vamos = np.array([109, 169, 228])
        PALLETE.cube = np.array([152, 155, 204])       

        PALLETE.left_arm = np.array([218, 119, 242])
        PALLETE.right_arm = np.array([151, 117, 250])
        PALLETE.left_leg = np.array([255, 212, 59])
        PALLETE.right_leg = np.array([255, 169, 77])

        PALLETE.head = np.array([134, 142, 150])
        PALLETE.body = np.array([134, 142, 150])

        # convert rgb to bgr
        for k, v in PALLETE.items():
            PALLETE[k] = v[::-1]
        return PALLETE
    
if __name__ == '__main__':
    vis =  FastVisualizer()

    image = '/github/Tennis.ai/assets/tempt_test.png'
    vis.set_image(image)
    np.random.seed(0)
    keypoints = np.random.randint(300, 600, (17, 2))
    from utils import Timer
    t= Timer()
    t.start()
    vis.draw_score_bar(0.94)
    # vis.draw_skeleton(keypoints)
    # vis.draw_non_transparent_area((0, 0, 100, 100), alpha=0.2)
    vis.draw_logo()
    cv2.imshow('test', vis.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()