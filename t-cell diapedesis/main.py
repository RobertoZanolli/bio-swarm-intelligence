
from sim import Simulation, SimulationView
if __name__ == "__main__":
    sim = Simulation(n_cells=30)
    view = SimulationView(sim)
    view.animate()