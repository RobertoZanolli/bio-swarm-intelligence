import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import random
from environment import VesselEnvironment
from agents import TCell, CellState

# ============================================================
# SIMULATION
# ============================================================

class Simulation:
    def __init__(self, n_cells=28):
        self.env = VesselEnvironment()
        self.cells = self._create_cells(n_cells)
        self.dt = 1.0
        self.noise_scale = 0.05

    def _remove_if_out_of_reach(self, cell: TCell) -> bool:
        at_downstream_end = cell.position[0] >= self.env.width - 1
        if not at_downstream_end:
            return False

        if cell.target_anchor is not None and cell.state == CellState.ADHERED:
            cell.target_anchor["occupied"] = max(0, cell.target_anchor["occupied"] - 1)

        cell.is_active = False
        cell.target_anchor = None
        cell.velocity[:] = 0.0
        return True

    def _create_cells(self, n: int):
        cells = []
        for _ in range(n):
            x = random.uniform(30, 180)
            y = random.uniform(self.env.vessel_top + 20, self.env.vessel_bottom - 20)
            vx = random.uniform(0.8, 1.8)
            vy = random.uniform(-0.08, 0.08)

            cell = TCell(
                position=np.array([x, y], dtype=float),
                velocity=np.array([vx, vy], dtype=float),
                adhesion_strength=random.uniform(0.8, 1.2),
                chemokine_sensitivity=random.uniform(0.85, 1.15),
                max_speed=random.uniform(2.4, 3.2),
            )
            cells.append(cell)
        return cells

    def repulsion_force(self, cell: TCell, radius: float = 20.0, strength: float = 0.16):
        force = np.zeros(2, dtype=float)

        for other in self.cells:
            if other is cell or not other.is_active:
                continue

            delta = cell.position - other.position
            dist = np.linalg.norm(delta)
            if 0 < dist < radius:
                direction = delta / dist
                force += direction * strength * (radius - dist) / radius

        return force

    def wall_confinement_force(self, cell: TCell, strength: float = 0.25):
        """
        Gently pushes flowing cells back into the lumen if they drift too close to walls.
        """
        y = cell.position[1]
        top = self.env.vessel_top
        bottom = self.env.vessel_bottom

        force = np.zeros(2, dtype=float)

        d_top = y - top
        d_bottom = bottom - y

        if d_top < 12:
            force[1] += strength
        if d_bottom < 12:
            force[1] -= strength

        return force

    def move_toward(self, src: np.ndarray, dst: np.ndarray, strength: float):
        delta = dst - src
        dist = np.linalg.norm(delta)
        if dist == 0:
            return np.zeros(2, dtype=float)
        return (delta / dist) * strength

    def update_flowing(self, cell: TCell):
        flow = self.env.flow_vector(cell.position)
        rep = self.repulsion_force(cell)
        wall_push = self.wall_confinement_force(cell)
        noise = np.random.normal(0, self.noise_scale, size=2)

        force = flow * 0.18 + rep + wall_push + noise
        cell.velocity += force
        cell.limit_speed()
        cell.position += cell.velocity * self.dt
        cell.position = self.env.clamp_inside_vessel(cell.position)

        if self._remove_if_out_of_reach(cell):
            return

        # Transition condition: if close enough to the wall, start rolling
        if self.env.near_wall(cell.position, threshold=16):
            cell.state = CellState.ROLLING
            cell.rolling_time = 0.0
            wall_name = self.env.nearest_wall_name(cell.position)
            anchor, _ = self.env.closest_anchor(cell.position, wall_name=wall_name)
            cell.target_anchor = anchor

    def update_rolling(self, cell: TCell):
        cell.rolling_time += self.dt

        wall_name = self.env.nearest_wall_name(cell.position)
        wall_y = self.env.nearest_wall_y(cell.position)
        anchor, anchor_dist = self.env.closest_anchor(cell.position, wall_name=wall_name)

        if anchor is not None:
            cell.target_anchor = anchor

        flow = self.env.flow_vector(cell.position) * cell.rolling_factor
        rep = self.repulsion_force(cell, radius=18, strength=0.12)
        noise = np.random.normal(0, self.noise_scale * 0.5, size=2)

        force = flow * 0.12 + rep + noise

        # Keep the cell near the current wall
        wall_delta = wall_y - cell.position[1]
        force[1] += 0.12 * wall_delta

        # Attraction to anchor if available
        if cell.target_anchor is not None:
            anchor_pos = cell.target_anchor["pos"]
            force += self.move_toward(cell.position, anchor_pos, strength=0.18)

        cell.velocity += force
        cell.limit_speed()

        # Rolling should be slow
        speed = np.linalg.norm(cell.velocity)
        rolling_max = 1.1
        if speed > rolling_max and speed > 0:
            cell.velocity = cell.velocity / speed * rolling_max

        cell.position += cell.velocity * self.dt
        cell.position = self.env.clamp_inside_vessel(cell.position)

        if self._remove_if_out_of_reach(cell):
            return

        # If it moves away from the wall too much, it detaches back to flowing
        if not self.env.near_wall(cell.position, threshold=20):
            cell.state = CellState.FLOWING
            cell.target_anchor = None
            return

        # Firm adhesion probability increases with:
        # - time spent rolling
        # - anchor strength
        # - chemokine signal
        # - cell adhesion strength
        if cell.target_anchor is not None:
            available = cell.target_anchor["occupied"] < cell.target_anchor["capacity"]

            chemokine = self.env.chemokine_signal(cell.position)
            arrest_score = (
                0.28 * min(cell.rolling_time / 18.0, 1.0)
                + 0.28 * min(cell.target_anchor["strength"], 1.0)
                + 0.22 * min(chemokine * cell.chemokine_sensitivity * 2.2, 1.0)
                + 0.22 * min(cell.adhesion_strength, 1.2)
            )

            close_enough = anchor_dist < 14.0

            if available and close_enough and arrest_score > 0.62:
                cell.state = CellState.ADHERED
                cell.velocity[:] = 0.0
                cell.target_anchor["occupied"] += 1

    def update_adhered(self, cell: TCell):
        if cell.target_anchor is not None:
            # Stick to the anchor position
            cell.position = cell.target_anchor["pos"].copy()

        chemokine = self.env.chemokine_signal(cell.position)
        progress_gain = 0.01 + 0.025 * chemokine * cell.chemokine_sensitivity
        cell.crossing_progress += progress_gain

        if cell.crossing_progress >= 0.35:
            cell.state = CellState.EXTRAVASATING

    def update_extravasating(self, cell: TCell):
        wall_name = self.env.nearest_wall_name(cell.position)

        # Move through the wall, outward
        direction = np.array([0.0, -1.0]) if wall_name == "top" else np.array([0.0, 1.0])
        drift_to_tissue = self.move_toward(
            cell.position,
            self.env.inflammation_center,
            strength=0.15
        )

        cell.position += direction * 1.8 + drift_to_tissue * 0.2
        cell.position = self.env.clamp_world(cell.position)
        cell.crossing_progress += 0.04

        crossed = (
            cell.position[1] < self.env.vessel_top - 8
            or cell.position[1] > self.env.vessel_bottom + 8
        )
        if crossed:
            cell.state = CellState.EXTRAVASATED
            if cell.target_anchor is not None:
                cell.target_anchor["occupied"] = max(0, cell.target_anchor["occupied"] - 1)

    def update_extravasated(self, cell: TCell):
        target_force = self.move_toward(
            cell.position,
            self.env.inflammation_center,
            strength=0.32 * cell.chemokine_sensitivity
        )
        rep = self.repulsion_force(cell, radius=18, strength=0.10)
        noise = np.random.normal(0, self.noise_scale * 0.35, size=2)

        cell.velocity += target_force + rep + noise

        # tissue movement should be moderate, not vessel-like
        speed = np.linalg.norm(cell.velocity)
        tissue_max = 1.7
        if speed > tissue_max and speed > 0:
            cell.velocity = cell.velocity / speed * tissue_max

        cell.position += cell.velocity * self.dt
        cell.position = self.env.clamp_world(cell.position)

    def step(self):
        for cell in self.cells:
            if not cell.is_active:
                continue

            if cell.state == CellState.FLOWING:
                self.update_flowing(cell)
            elif cell.state == CellState.ROLLING:
                self.update_rolling(cell)
            elif cell.state == CellState.ADHERED:
                self.update_adhered(cell)
            elif cell.state == CellState.EXTRAVASATING:
                self.update_extravasating(cell)
            elif cell.state == CellState.EXTRAVASATED:
                self.update_extravasated(cell)

        self.cells = [cell for cell in self.cells if cell.is_active]


# ============================================================
# VISUALIZATION
# ============================================================

class SimulationView:
    def __init__(self, sim: Simulation):
        self.sim = sim
        self.fig, self.ax = plt.subplots(figsize=(13, 7))
        self.anim = None

    def draw_environment(self):
        env = self.sim.env

        self.ax.clear()
        self.ax.set_xlim(0, env.width)
        self.ax.set_ylim(env.height, 0)  # invert y-axis for screen-like view
        self.ax.set_aspect("equal", adjustable="box")

        # Vessel walls
        self.ax.plot([0, env.width], [env.vessel_top, env.vessel_top], linewidth=2)
        self.ax.plot([0, env.width], [env.vessel_bottom, env.vessel_bottom], linewidth=2)

        # Inflammatory region
        circle = Circle(
            tuple(env.inflammation_center),
            env.inflammation_radius,
            fill=False,
            linewidth=2,
            linestyle="--",
        )
        self.ax.add_patch(circle)
        self.ax.text(
            env.inflammation_center[0] - 60,
            env.inflammation_center[1] - 60,
            "Inflammatory site",
            fontsize=10,
        )

        # Anchors
        for anchor in env.anchors:
            pos = anchor["pos"]
            label = f"{anchor['occupied']}/{anchor['capacity']}"
            self.ax.scatter(pos[0], pos[1], marker="x", s=70)
            self.ax.text(pos[0] + 8, pos[1] - 8, label, fontsize=8)

        # Region labels
        self.ax.text(20, env.vessel_top - 20, "Tissue", fontsize=11)
        self.ax.text(20, (env.vessel_top + env.vessel_bottom) / 2, "Vessel lumen", fontsize=11)
        self.ax.text(20, env.vessel_bottom + 30, "Tissue", fontsize=11)

        # Flow arrow text
        self.ax.text(env.width - 180, (env.vessel_top + env.vessel_bottom) / 2 - 20, "Blood flow →", fontsize=11)

    def draw_cells(self):
        state_to_marker = {
            CellState.FLOWING: "o",
            CellState.ROLLING: "s",
            CellState.ADHERED: "^",
            CellState.EXTRAVASATING: "D",
            CellState.EXTRAVASATED: "P",
        }

        state_to_size = {
            CellState.FLOWING: 45,
            CellState.ROLLING: 50,
            CellState.ADHERED: 65,
            CellState.EXTRAVASATING: 60,
            CellState.EXTRAVASATED: 55,
        }

        for cell in self.sim.cells:
            self.ax.scatter(
                cell.position[0],
                cell.position[1],
                marker=state_to_marker[cell.state],
                s=state_to_size[cell.state],
            )

    def draw_legend_text(self):
        lines = [
            "States:",
            "o  FLOWING",
            "s  ROLLING",
            "^  ADHERED",
            "D  EXTRAVASATING",
            "P  EXTRAVASATED",
            "x  anchor / adhesion site",
        ]
        x0 = 15
        y0 = 20
        for i, line in enumerate(lines):
            self.ax.text(x0, y0 + i * 16, line, fontsize=9)

        counts = {state: 0 for state in CellState}
        for cell in self.sim.cells:
            counts[cell.state] += 1

        summary = (
            f"Flowing: {counts[CellState.FLOWING]}   "
            f"Rolling: {counts[CellState.ROLLING]}   "
            f"Adhered: {counts[CellState.ADHERED]}   "
            f"Extravasating: {counts[CellState.EXTRAVASATING]}   "
            f"Extravasated: {counts[CellState.EXTRAVASATED]}"
        )
        self.ax.set_title("T-Cell Diapedesis Simulation\n" + summary)

    def update(self, _frame):
        self.sim.step()
        self.draw_environment()
        self.draw_cells()
        self.draw_legend_text()

    def animate(self):
        self.anim = FuncAnimation(self.fig, self.update, interval=50, cache_frame_data=False)
        plt.show()

