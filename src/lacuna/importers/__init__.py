"""Convert compatible trained models into ordinary Lacuna networks.

The layer builders are framework neutral. The SLAYER adapter loads its
optional training dependencies only when conversion or validation is called.
"""

from .dense_lif import DenseLIFDeployment, DenseLIFLayer, build_dense_lif
from .feedforward_lif import (
    Conv2dLIFLayer,
    FeedforwardLIFDeployment,
    build_feedforward_lif,
)
from .slayer import (
    SlayerDenseImport,
    SlayerFeedforwardImport,
    import_slayer_dense,
    import_slayer_feedforward,
    validate_slayer_dense,
    validate_slayer_feedforward,
)

__all__ = [
    "Conv2dLIFLayer",
    "DenseLIFDeployment",
    "DenseLIFLayer",
    "FeedforwardLIFDeployment",
    "SlayerDenseImport",
    "SlayerFeedforwardImport",
    "build_dense_lif",
    "build_feedforward_lif",
    "import_slayer_dense",
    "import_slayer_feedforward",
    "validate_slayer_dense",
    "validate_slayer_feedforward",
]
