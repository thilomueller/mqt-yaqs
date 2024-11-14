from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.data_structures.MPO import MPO
    from yaqs.data_structures.MPS import MPS

from yaqs.methods.TDVP_Mendl import single_site_TDVP


def dynamic_TDVP(state: 'MPS', H: 'MPO', dt: float, max_bond_dim: int):
    current_max_bond_dim = state.read_max_bond_dim()

    # if current_max_bond_dim < max_bond_dim:
    #     return False
    #     # Perform 2TDVP
    #     # integrate_local_twosite_modified(H, stochastic_MPS, dt*1j, numsteps=1, numiter_lanczos=25, tol_split=1e-6, max_bond_dim=max_bond_dim)
    #     # return two_site_TDVP(state, H, dt, max_bond_dim)
    # else:
        # Perform 1TDVP
    single_site_TDVP(H, state, dt, numsteps=1)

