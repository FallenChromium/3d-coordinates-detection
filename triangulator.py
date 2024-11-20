from itertools import permutations

import numpy as np
from convertor import CameraWorldConvertor


class Triangulator:
    def __init__(self, convertors: list[CameraWorldConvertor]):
        self.convertors = convertors

    def triangulate(self, points: np.ndarray): 
        lines = []
        print(points)
        for conventor, point in zip(self.convertors, points):
            print(conventor, point)
            if point is not None:
                line = conventor.transform_point_to_world_coord(point)
                lines.append(line)
        
        if len(lines) < 2:
            return None
        
        points = []
        for id1, line1 in enumerate(lines):
            for id2, line2 in enumerate(lines):
                if id1 != id2:
                    point1 = line1[1]
                    point2 = line2[1]
                    vec1 = line1[0][:3]
                    vec2 = line2[0][:3]

                    n = np.cross(vec1, vec2)
                    n2 = np.cross(vec2, n)

                    c1 = point1 + (((point2 - point1) @ n2) / (vec1 @ n2)) * vec1
                    points.append(c1)

        return np.mean(points, axis=0)


class SingleCameraTriangulator:
    def __init__(self, convertor: CameraWorldConvertor, object_diameter: float):
        self.object_diameter = object_diameter
        self.convertor = convertor

    #list of opposite pairs of obj
    def triangulate(self, points_pairs: list[tuple[np.ndarray, np.ndarray]]): 
        positions = []
        for point1, point2 in points_pairs:
            if point1 is not None and point2 is not None:
                line1 = self.convertor.transform_point_to_world_coord(point1)
                line2 = self.convertor.transform_point_to_world_coord(point2)

                vec1 = line1[0][:3]
                vec2 = line2[0][:3]

                norm_vec1 = vec1 / np.linalg.norm(vec1)
                norm_vec2 = vec2 / np.linalg.norm(vec2)

                dist = self.object_diameter / (np.linalg.norm(norm_vec1 - norm_vec2))

                central_vector = (norm_vec1 + norm_vec2) 
                norm_central_vector = central_vector / np.linalg.norm(central_vector)
                
                positions.append(line1[1] + dist * norm_central_vector)
                
        return np.mean(positions, axis=0)
    


if __name__ == "__main__":
    FOCAL_LENGHT = 0.035
    CAMERA1_POS = np.array([660, 760, 35])
    CAMERA2_POS = np.array([810, 740, 45])
    CAMERA3_POS = np.array([900, -400, 80])
    CAMERA1_AZIMUTH = -110
    CAMERA2_AZIMUTH = -125
    CAMERA3_AZIMUTH = -225
    MATRIX_PARAMS = np.array([23.760, 13.365]) / 1000
    IMG_PARAMS = np.array([1920, 1080])


    t = Triangulator(
        [CameraWorldConvertor(CAMERA1_POS, CAMERA1_AZIMUTH, FOCAL_LENGHT, IMG_PARAMS, MATRIX_PARAMS),
         CameraWorldConvertor(CAMERA2_POS, CAMERA2_AZIMUTH, FOCAL_LENGHT, IMG_PARAMS, MATRIX_PARAMS),
         CameraWorldConvertor(CAMERA3_POS, CAMERA3_AZIMUTH, FOCAL_LENGHT, IMG_PARAMS, MATRIX_PARAMS)])
    point = t.triangulate(points = [np.array([192.48530412, -195.27369487]), np.array([-48.61184182, -237.59277193]), None])
    print(point)

