class Primitive:
    def __init__(self) -> None:
        self.type = None

    def __call__(self, robot, action) -> bool:
        """Once the derived primitive is called it will execute the motion defined in each secial case.
        """
        raise NotImplementedError(
            "Primitive class can not be called directly!")
