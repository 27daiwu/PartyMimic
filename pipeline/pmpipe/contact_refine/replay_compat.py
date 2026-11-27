"""thin re-export of the scene helpers of pmpipe.replay (kept separate so contact_refine does not import replay's heavy module-level state twice)."""
from __future__ import annotations

import numpy as np

from ..replay import FONT, PH, PW, _scene_model as scene_model


def marker(scn, kind, pos, size, rgba, connector=None):
    import mujoco
    if scn.ngeom >= scn.maxgeom: return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.array(size, float), np.array(pos, float), np.eye(3).flatten(), np.array(rgba, np.float32))
    if connector is not None: mujoco.mjv_connector(g, kind, float(size[0]), np.array(connector[0], float), np.array(connector[1], float))
    scn.ngeom += 1
