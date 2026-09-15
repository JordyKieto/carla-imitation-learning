import carla

class EgoContext:
    """
    Shared context for cooperative multiple inheritance.
    Sets self.world and self.vehicle exactly once.
    """

    def __init__(self, *args, world: carla.World, vehicle: carla.Vehicle, **kwargs):
        self.world = world
        self.vehicle = vehicle
        super().__init__(*args, **kwargs)