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


def show_vehicle_distance(img, box_result, lanes, main_lane):
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    vehicles = np.vstack(box_result)
    vehicles = vehicles[vehicles[:, -1] >= 0.5]

    bottom_centers = vehicles[:, 2:4]
    bottom_centers[:, 0] -= 0.5 * (vehicles[:, 2] - vehicles[:, 0])

    inside_main_lane = np.hstack(
        [point_in_polygon(bc, lanes[main_lane]) for bc in bottom_centers])

    inside_bottom_centers = bottom_centers[inside_main_lane]

    closest_inside_bottom_centers = inside_bottom_centers[
        inside_bottom_centers[:, 1] == inside_bottom_centers[:, 1].max()]

    for ibc in closest_inside_bottom_centers:
        draw.line([(ibc[0], ibc[1]), (ibc[0], h)],
                  fill=(255, 255, 255), width=2)
    return img
