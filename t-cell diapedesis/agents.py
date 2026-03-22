import numpy as np
from enum import Enum, auto


# ============================================================
# STATES
# ============================================================

class CellState(Enum):
    FLOWING = auto()
    ROLLING = auto()
    ADHERED = auto()
    EXTRAVASATING = auto()
    EXTRAVASATED = auto()


# ============================================================
# AGENT
# ============================================================

class TCell:
    def __init__(self, position, velocity, max_speed=2.0, adhesion_strength=0.8, chemokine_sensitivity=1.0, rolling_factor=0.3):
        self.position = position
        self.velocity = velocity
        self.state = CellState.FLOWING
        self.max_speed = max_speed
        self.adhesion_strength = adhesion_strength
        self.chemokine_sensitivity = chemokine_sensitivity
        self.rolling_factor = rolling_factor
        self.rolling_time = 0.0
        self.is_active = True
        self.target_anchor = None
        self.crossing_progress = 0.0
        self.radius = 5.0



    def limit_speed(self):
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed and speed > 0:
            self.velocity = self.velocity / speed * self.max_speed

