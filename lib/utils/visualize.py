from PIL import Image, ImageDraw

import numpy as np


def show_lanes(img, lanes, main_lane=None):
    """
    Args:
        img (str or Image.Image or numpy.ndarray): image to draw
        lanes (List[numpy.ndarray]): lanes represented by polygons
        main_lane (int): index of main lane. Default to None.
    Returns:
        Image.Image: painted image
    """
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    w, h = img.size
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    draw = ImageDraw.Draw(img)

    for i, lane in enumerate(lanes):
        draw.polygon(lanes[i].flatten().tolist(), outline=colors[i])

    if main_lane:
        draw.text((lanes[main_lane].mean(0)[0], lanes[main_lane].mean(0)[1]), text='Main')

    arrow_polygon = [w / 2 - 5, h, w / 2 + 5, h,
                    w / 2 + 5, h - 20, w / 2 + 10, h - 20,
                    w / 2, h - 30,
                    w / 2 - 10, h - 20, w / 2 - 5, h - 20,
                    w / 2 - 5, h]
    draw.polygon(arrow_polygon, fill='white')
    return img
