class SimulationParams:
    def __init__(self, measurements: dict[str, int], T: float, dt: float, max_bond_dim: int, N: int):
        self.measurements = measurements
        self.T = T
        self.dt = dt
        self.max_bond_dim = max_bond_dim
        self.N = N
