from typing import TYPE_CHECKING

from genesis.typing import PositiveFloat

from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.dem_entity import DEMEntity


class Sand(Base["DEMEntity"]):
    """
    The sand material class for DEM.

    The defaults follow the reference implementation (`sand-water-coupling-PIC-DEM-3d`): the grain stiffness
    (`young_modulus`) and density (`rho`) enter the contact model only through the ratio `young / rho`, so the
    reference values are kept verbatim to reproduce its stiffness-to-gravity ratio exactly.

    Parameters
    ----------
    rho : float, optional
        The density of the sand grain. Defaults to 2.5, matching the reference implementation.
    young_modulus : float, optional
        Young's modulus of the sand grain. The normal contact stiffness is `young_modulus * radius`.
        Defaults to 1e6, matching the reference implementation.
    poisson_ratio : float, optional
        Poisson ratio of the sand grain. The tangential contact stiffness is `young * radius * poisson_ratio`.
        Defaults to 0.3, matching the reference implementation.
    friction_angle : float, optional
        Friction angle in radians. Tangential contact forces are clamped by the Coulomb limit
        `|f_shear| <= |f_normal| * tan(friction_angle)`. Defaults to 0.5, matching the reference implementation.
    """

    rho: PositiveFloat = 2.5
    young_modulus: PositiveFloat = 1e6
    poisson_ratio: PositiveFloat = 0.3
    friction_angle: PositiveFloat = 0.5
