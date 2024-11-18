class SimulationParams:
    def __init__(self,  T: float, dt: float, max_bond_dim: int, N: int):
        self.T = T
        self.dt = dt
        self.max_bond_dim = max_bond_dim
        self.N = N