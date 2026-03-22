
"""
                    TISSUE / INFLAMMATORY SITE
        -------------------------------------------------
                         *  *  *  *  *
                         chemokine gradient

=============================================================  ← endothelial wall
|                                                           |
|   o   o    o        o      o        o        o            |   → blood flow direction
|                                                           |
|      o        o        o         o                        |
|                                                           |
|  o         o        o            o                        |
|                                                           |
=============================================================  ← endothelial wall

legend:
o = t cell
* = inflammatory signals / chemokines
"""

import numpy as np

class VesselEnvironment:

    def __init__(self, width=1000, height=600):
        self.width = width
        self.height = height

        self.vessel_top = 220
        self.vessel_bottom = 380

        self.inflammation_center = np.array([850.0, 500.0])
        self.inflammation_radius = 50.0

        self.anchors = self._create_anchors()

    def _create_anchors(self):
        xs = np.linspace(150, 850, 8)
        anchors = []

        for i, x in enumerate(xs):
            y = self.vessel_top if i % 2 == 0 else self.vessel_bottom
            anchors.append({
                "pos": np.array([x, y], dtype=float),
                "strength": 1.0,
                "capacity": 2,
                "occupied": 0,
            })

        return anchors

    def in_vessel(self, pos):
        x, y = pos
        return 0 <= x <= self.width and self.vessel_top < y < self.vessel_bottom

    def in_tissue(self, pos):
        x, y = pos
        return 0 <= x <= self.width and (y <= self.vessel_top or y >= self.vessel_bottom)

    def near_wall(self, pos, threshold=15):
        _, y = pos
        return (
            abs(y - self.vessel_top) < threshold or
            abs(y - self.vessel_bottom) < threshold
        )

    def nearest_wall(self, pos):
        _, y = pos
        d_top = abs(y - self.vessel_top)
        d_bottom = abs(y - self.vessel_bottom)
        return "top" if d_top < d_bottom else "bottom"

    def nearest_wall_name(self, pos):
        return self.nearest_wall(pos)

    def nearest_wall_y(self, pos):
        return self.vessel_top if self.nearest_wall(pos) == "top" else self.vessel_bottom

    def flow_vector(self, pos):
        _, y = pos
        if not self.in_vessel(pos):
            return np.zeros(2)

        center = (self.vessel_top + self.vessel_bottom) / 2
        half_height = (self.vessel_bottom - self.vessel_top) / 2
        r = (y - center) / half_height
        speed = 2.0 * (1 - r**2)
        return np.array([max(speed, 0.0), 0.0])

    def chemokine_signal(self, pos):
        distance = np.linalg.norm(pos - self.inflammation_center)
        return np.exp(-distance / 120.0)

    def closest_anchor(self, pos, wall_name=None):
        best_anchor = None
        best_dist = float("inf")

        for anchor in self.anchors:
            if wall_name is not None:
                anchor_wall = "top" if abs(anchor["pos"][1] - self.vessel_top) < abs(anchor["pos"][1] - self.vessel_bottom) else "bottom"
                if anchor_wall != wall_name:
                    continue

            d = np.linalg.norm(pos - anchor["pos"])
            if d < best_dist:
                best_dist = d
                best_anchor = anchor

        return best_anchor, best_dist

    def clamp_inside_vessel(self, pos):
        x = np.clip(pos[0], 0, self.width)
        y = np.clip(pos[1], self.vessel_top + 1, self.vessel_bottom - 1)
        return np.array([x, y], dtype=float)

    def clamp_world(self, pos):
        x = np.clip(pos[0], 0, self.width)
        y = np.clip(pos[1], 0, self.height)
        return np.array([x, y], dtype=float)