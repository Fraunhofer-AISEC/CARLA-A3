import numpy as np
import carla

class CameraGeometry:

    def __init__(self, camResolutionX, camResolutionY, camFOV) -> None:
        self.width = camResolutionX
        self.height = camResolutionY
        self.fov = camFOV
        self.projectionMatrix = self.build_projection_matrix(self.width, self.height, self.fov)

    @staticmethod
    def build_projection_matrix(width, height, fov):
        width = int(width)
        height = int(height)
        fov = float(fov)
        
        focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = width / 2.0
        K[1, 2] = height / 2.0
        return K
    
    @staticmethod
    def get_image_point(
        loc, # location
        K,   # project matrix
        w2c  # world to camera projection matrix
    ):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

    def bounding_box_to_pixel_coordinate(
        self,
        boundingBox : carla.BoundingBox,
        dashCam : carla.ServerSideSensor,
        distanceThreshold : float,
    ):
        dashCamTransform = dashCam.get_transform()
        forwardVec = dashCamTransform.get_forward_vector()
        ray = boundingBox.location - dashCamTransform.location

        (x_min, y_min) = (x_max, y_max) = (None, None)
        if (
            forwardVec.dot(ray) > 1 and
            boundingBox.location.distance(dashCamTransform.location) < distanceThreshold
        ):

            vertices = boundingBox.get_world_vertices(carla.Transform())
            pixelPoints = np.array(
                [
                    self.get_image_point(
                        vertex_location, 
                        self.projectionMatrix,
                        dashCamTransform.get_inverse_matrix()
                    ) for vertex_location in vertices
                ]
            )

            (x_min, y_min) = np.min(pixelPoints, 0).astype(np.int)
            (x_max, y_max) = np.max(pixelPoints, 0).astype(np.int)

        return (x_min, y_min), (x_max, y_max)