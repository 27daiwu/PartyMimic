"""PartyMimic contact_refine: dance-independent contact identification, ground calibration, planted-foot constraints, root-XYZ optimisation and bounded leg IK for
video-derived 50 Hz reference motions, with a pre-registered data-QA gate.  CLI: `./pm contact-refine <dance> [--analyze-only]`, `./pm contact-qa <dance>`.
Layers: geometry (g1mj FK / Jacobians) -> ground -> contacts -> solver -> pipeline -> export / metrics -> qa -> api (+ diagnostics / viz / video / report)."""
from .config import GATES_VERSION, QA_GATES, VERSION, Params  # noqa: F401
