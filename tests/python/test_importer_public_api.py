from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import lacuna.importers as importers


PUBLIC_EXPORTS = {
    "Conv2dLIFLayer": "feedforward_lif",
    "DenseLIFDeployment": "dense_lif",
    "DenseLIFLayer": "dense_lif",
    "FeedforwardLIFDeployment": "feedforward_lif",
    "SlayerDenseImport": "slayer",
    "SlayerFeedforwardImport": "slayer",
    "build_dense_lif": "dense_lif",
    "build_feedforward_lif": "feedforward_lif",
    "import_slayer_dense": "slayer",
    "import_slayer_feedforward": "slayer",
    "validate_slayer_dense": "slayer",
    "validate_slayer_feedforward": "slayer",
}


def test_importer_public_exports_are_declared() -> None:
    assert set(importers.__all__) == set(PUBLIC_EXPORTS)
    assert len(importers.__all__) == len(PUBLIC_EXPORTS)


@pytest.mark.parametrize("name,module_name", PUBLIC_EXPORTS.items())
def test_importer_public_exports_match_submodule_definitions(
    name: str, module_name: str
) -> None:
    module = importlib.import_module(f"lacuna.importers.{module_name}")
    assert getattr(importers, name) is getattr(module, name)


def test_public_imports_and_graph_building_do_not_import_torch_or_lava(
    tmp_path: Path,
) -> None:
    source_path = Path(__file__).resolve().parents[2] / "src"
    script = textwrap.dedent(
        """
        import importlib.abc
        import json
        from pathlib import Path
        import sys

        def optional_framework_loaded():
            return any(
                name.split(".", 1)[0] in {"torch", "lava"}
                for name in sys.modules
            )

        assert not optional_framework_loaded()
        attempted_imports = []

        class RejectOptionalFrameworks(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in {"torch", "lava"}:
                    attempted_imports.append(fullname)
                    raise ModuleNotFoundError(
                        f"Optional framework import is forbidden: {fullname}",
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, RejectOptionalFrameworks())
        sys.path.insert(0, sys.argv[1])

        import lacuna
        import lacuna.importers
        from lacuna.importers import (
            Conv2dLIFLayer,
            DenseLIFDeployment,
            DenseLIFLayer,
            FeedforwardLIFDeployment,
            SlayerDenseImport,
            SlayerFeedforwardImport,
            build_dense_lif,
            build_feedforward_lif,
            import_slayer_dense,
            import_slayer_feedforward,
            validate_slayer_dense,
            validate_slayer_feedforward,
        )

        dense = build_dense_lif(
            [DenseLIFLayer([[1, -2], [-3, 4]], 2, 1)]
        )
        assert isinstance(dense, DenseLIFDeployment)
        assert dense.layer_nodes == ((2, 3),)
        assert {
            (edge.pre, edge.post): edge.weight
            for edge in dense.network.graph.edges
        } == {(0, 2): 1, (1, 2): -2, (0, 3): -3, (1, 3): 4}

        spatial = build_feedforward_lif(
            [
                Conv2dLIFLayer([[[[1, -2]]]], 2, 1),
                DenseLIFLayer([[3, -4]], 2, 1),
            ],
            (1, 1, 3),
        )
        assert isinstance(spatial, FeedforwardLIFDeployment)
        assert spatial.input_shape == (1, 1, 3)
        assert spatial.layer_shapes == ((1, 1, 2), (1,))
        assert spatial.layer_nodes == ((3, 4), (5,))
        assert {
            (edge.pre, edge.post): edge.weight
            for edge in spatial.network.graph.edges
        } == {
            (0, 3): 1,
            (1, 3): -2,
            (1, 4): 1,
            (2, 4): -2,
            (3, 5): 3,
            (4, 5): -4,
        }

        # Build and serialize ordinary graphs without native execution.
        for name, deployment in (("dense", dense), ("spatial", spatial)):
            graph = deployment.network.graph
            assert all(
                node.polarity is lacuna.NeuronPolarity.MIXED
                for node in graph.nodes
            )
            assert all(
                edge.plasticity is None and edge.weight_group is None
                for edge in graph.edges
            )
            path = Path(sys.argv[2]) / f"{name}.json"
            deployment.network.save(path)
            loaded = lacuna.Network.load(path)
            assert loaded.graph.to_text() == graph.to_text()
            assert (
                loaded.semantic_sha256 == deployment.network.semantic_sha256
            )

        assert attempted_imports == [], attempted_imports
        assert not optional_framework_loaded()
        print(json.dumps({"graphs": 2, "optional_import_attempts": 0}))
        """
    )
    process = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source_path), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(process.stdout) == {
        "graphs": 2,
        "optional_import_attempts": 0,
    }
