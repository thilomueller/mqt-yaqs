from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.data_structures.MPO import MPO
    from yaqs.data_structures.MPS import MPS

from yaqs.methods.TDVP import single_site_TDVP, two_site_TDVP


def dynamic_TDVP(state: 'MPS', H: 'MPO', dt: float, max_bond_dim: int):
    current_max_bond_dim = state.write_max_bond_dim()

    # single_site_TDVP(state, H, dt, numsteps=1)
    if current_max_bond_dim <= max_bond_dim:
        # Perform 2TDVP
        two_site_TDVP(state, H, dt, numsteps=1, tol_split=1e-6)
    else:
        # Perform 1TDVP
        single_site_TDVP(state, H, dt, numsteps=1)

