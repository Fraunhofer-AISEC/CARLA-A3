class MethodNotImplementedError(Exception):
    pass

class ObjectDetectorBase:
    def __init__(self) -> None:
        pass

    def detect(self, *input) -> None:
        raise MethodNotImplementedError(
            f"Module [{type(self).__name__}] is missing the required \"detect\" function"
        )