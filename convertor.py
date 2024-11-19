import numpy as np


class CameraWorldConvertor:
    def __init__(self, CAMERA_POS: np.ndarray,
                 CAMERA_AZIMUTH: float,
                 FOCAL_LENGHT: float,
                 IMG_PARAMS: np.ndarray,
                 MATRIX_PARAMS: np.ndarray):
        self.CAMERA_POS = CAMERA_POS
        self.CAMERA_AZIMUTH = CAMERA_AZIMUTH
        self.FOCAL_LENGHT = FOCAL_LENGHT
        self.IMG_PARAMS = IMG_PARAMS
        self.MATRIX_PARAMS = MATRIX_PARAMS
        self.__build_matrices()

    def __build_matrices(self):
        self.axis_change_matrix = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        azimuth = self.CAMERA_AZIMUTH / 180 * np.pi
        rotation_matrix = np.array([
            [np.cos(azimuth), -np.sin(azimuth), 0, 0],
            [np.sin(azimuth), np.cos(azimuth), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]).T
        back_azimuth = azimuth * -1
        back_rotation_matrix = np.array([
            [np.cos(back_azimuth), -np.sin(back_azimuth), 0, 0],
            [np.sin(back_azimuth), np.cos(back_azimuth), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]).T
        movement_matrix = np.array([[1, 0, 0, -self.CAMERA_POS[0]],
                                    [0, 1, 0, -self.CAMERA_POS[1]],
                                    [0, 0, 1, -self.CAMERA_POS[2]],
                                    [0, 0, 0, 1]])
        back_movement_matrix = np.array([[1, 0, 0, self.CAMERA_POS[0]],
                                         [0, 1, 0, self.CAMERA_POS[1]],
                                         [0, 0, 1, self.CAMERA_POS[2]],
                                         [0, 0, 0, 1]])

        self.transform_to_img_matrix = rotation_matrix @ movement_matrix
        self.transform_to_world_matrix = back_movement_matrix @ back_rotation_matrix

        self.perspective_matrix = np.array([
            [self.FOCAL_LENGHT, 0, 0, 0],
            [0, self.FOCAL_LENGHT, 0, 0],
            [0, 0, 1, 0],
        ])

    def transform_point_to_img_coord(self, point: np.ndarray):
        CAMERA_POINT = self.transform_to_img_matrix @ point
        MATRIX_COORDS = self.perspective_matrix @ (self.axis_change_matrix @ CAMERA_POINT) / CAMERA_POINT[0]

        return MATRIX_COORDS[:2]/self.MATRIX_PARAMS*self.IMG_PARAMS

    def transform_point_to_world_coord(self, point: np.ndarray):
        MATRIX_POINT = np.array([*(point/self.IMG_PARAMS*self.MATRIX_PARAMS), 1.],dtype = np.float32)
        vector = [1., MATRIX_POINT[0]/self.FOCAL_LENGHT, MATRIX_POINT[1]/self.FOCAL_LENGHT, 1.]
        vec_world = self.transform_to_world_matrix @ vector - \
            self.transform_to_world_matrix @ np.array([0, 0, 0, 1]).T

        return (vec_world, point)


if __name__ == "__main__":
    conventor = CameraWorldConvertor(
        CAMERA_POS=np.array([660, 760, 35]),
        CAMERA_AZIMUTH=-110, MATRIX_PARAMS=np.array([23.760, 13.365]) / 1000, IMG_PARAMS=np.array([1920, 1080]),
        FOCAL_LENGHT=0.035)
    print(conventor.transform_point_to_img_coord(np.array(np.array([551.38,	383.84,	8.03, 1]).T)))
    line = conventor.transform_point_to_world_coord(conventor.transform_point_to_img_coord(
        np.array(np.array([551.38, 383.84, 8.03, 1]).T)))
    print(line)
