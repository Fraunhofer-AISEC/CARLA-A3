import carla
import numpy as np

class Image:

    def __init__(
        self,
        colorConverter: carla.ColorConverter = carla.ColorConverter.Raw
    ) -> None:
        self.colorConverter : carla.ColorConverter = colorConverter
        self.image : carla.SensorData = None

    def updateImage(self, image : carla.SensorData):
        self.image = image

        if self.image is None:
            return
        if self.colorConverter == carla.ColorConverter.Raw:
            return

        self.image.convert(self.colorConverter)

    def toNumpy(self):
        array = np.empty((800,800), dtype=np.uint8)
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
        return np.ascontiguousarray(array, dtype=np.uint8)


class LidarImage:
    def __init__(self) -> None:
        self.image : carla.LidarMeasurement = None

    def updateImage(self, image:carla.LidarMeasurement):
        self.image = image

    def toNumpy(self):
        pass