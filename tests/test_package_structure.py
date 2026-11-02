"""Public/compatibility wiring and supported-versus-legacy package boundaries."""
import importlib
import importlib.util
import unittest
from dataclasses import fields
from types import SimpleNamespace

import torch

import neural_ssm
from neural_ssm import DeepSSM, SSMConfig
from neural_ssm.ssm.registry import (
    SSM_PARAMETRIZATIONS, LTI_PARAMETRIZATIONS, SELECTIVE_PARAMETRIZATIONS,
    LEGACY_PARAMETRIZATIONS,
)


class PackageStructureTests(unittest.TestCase):
    def test_compatibility_imports_resolve_to_canonical_objects(self):
        cases = (
            ("ssm.deep", "ssm.layers", "DeepSSM"),
            ("ssm.layers", "ssm.config", "SSMConfig"),
            ("ssm.layers", "ssm.registry", "SSMParametrization"),
            ("ssm.lti_cells", "ssm.cells.lti.lru", "LRU"),
            ("ssm.lti_cells", "ssm.cells.lti.l2n", "Block2x2DenseL2SSM"),
            ("ssm.lti_cells", "ssm.cells.legacy.l2ru", "L2RU"),
            ("ssm.lti_cells", "ssm.cells.legacy.lruz", "lruz"),
            ("ssm.lti_cells", "ssm.cells.legacy.l2nt", "L2BoundedLTICell"),
            ("ssm.defect", "ssm.cells.lti.defect", "DefectL2SSM"),
            ("ssm.selective_cells", "ssm.cells.selective.tv", "RobustMambaDiagSSM"),
            ("ssm.selective_cells", "ssm.cells.selective.tvc", "RobustMambaDiagLTI"),
            ("ssm.scan_utils", "utils.scan", "compute_block_lti_states"),
            ("ssm.cache_utils", "utils.cache", "EvalCacheMixin"),
            ("ssm.state_utils", "utils.state", "resolve_runtime_state"),
            ("static_layers.config", "static_layers.generic_layers", "LayerConfig"),
            ("static_layers.linear", "static_layers.lipschitz_mlps", "L2BoundedLinearExact"),
            ("ssm.contextual.model", "ssm.contextual", "ContextualDeepSSM"),
            ("utils.cache", "utils.runtime", "EvalCacheMixin"),
        )
        for old, new, name in cases:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(importlib.import_module("neural_ssm." + old), name),
                    getattr(importlib.import_module("neural_ssm." + new), name),
                )

    def test_static_layer_sources_separate_ordinary_and_bounded_constructions(self):
        from neural_ssm.static_layers import (
            GLU, MLP, TLIP, LMLP, L2BoundedLinearExact, L2BoundedGLUv2,
            BudgetedL2BoundedGLUv2, MultiBranchLipMixer,
        )
        for cls in (GLU, MLP):
            self.assertEqual(cls.__module__, "neural_ssm.static_layers.generic_layers")
        for cls in (TLIP, LMLP, L2BoundedLinearExact, L2BoundedGLUv2,
                    BudgetedL2BoundedGLUv2, MultiBranchLipMixer):
            self.assertEqual(cls.__module__, "neural_ssm.static_layers.lipschitz_mlps")

    def test_family_registries_match_implementations_and_keep_legacy_wired(self):
        self.assertEqual(set(LTI_PARAMETRIZATIONS), {"lru", "l2n", "defect"})
        self.assertEqual(set(SELECTIVE_PARAMETRIZATIONS), {"tv", "tvc"})
        self.assertEqual(set(LEGACY_PARAMETRIZATIONS), {"l2ru", "zak", "l2nt"})
        self.assertEqual(set(SSM_PARAMETRIZATIONS),
                         set(LTI_PARAMETRIZATIONS) | set(SELECTIVE_PARAMETRIZATIONS) | set(LEGACY_PARAMETRIZATIONS))
        for name, spec in SSM_PARAMETRIZATIONS.items():
            with self.subTest(param=name):
                model = DeepSSM(2, 2, d_model=4, d_state=4, n_layers=1, param=name,
                                gamma=None if name == "lru" else 1.)
                cell = model.blocks[0].lru
                family = "legacy" if spec.legacy else spec.family
                self.assertTrue(type(cell).__module__.startswith("neural_ssm.ssm.cells." + family + "."))
                self.assertEqual(spec.legacy, name in LEGACY_PARAMETRIZATIONS)

    def test_removed_components_have_no_modules_exports_or_config_fields(self):
        for path in ("ssm.certified_transformer", "ssm.stable_recurrent_transformer",
                     "experimental.raven", "experimental.transformers"):
            self.assertIsNone(importlib.util.find_spec("neural_ssm." + path))
        for name in ("MultiHeadRavenRSM", "L2SelectiveRavenCell", "CertifiedTransformer", "StableRecurrentTransformer"):
            self.assertNotIn(name, neural_ssm.__all__)
            self.assertFalse(hasattr(neural_ssm, name))
        self.assertFalse(any(f.name.startswith("raven_") for f in fields(SSMConfig)))
        with self.assertRaisesRegex(ValueError, "Unknown SSM parametrization"):
            DeepSSM(2, 2, param="raven")

    def test_checkpoint_roundtrip_has_no_module_path_or_wrapper_prefix_changes(self):
        for param in SSM_PARAMETRIZATIONS:
            with self.subTest(param=param):
                kwargs = dict(d_model=4, d_state=4, n_layers=1, param=param,
                              gamma=None if param == "lru" else 1.)
                model = DeepSSM(2, 2, **kwargs).eval()
                restored = DeepSSM(2, 2, **kwargs).eval()
                checkpoint = model.state_dict()
                restored.load_state_dict(checkpoint, strict=True)
                self.assertEqual(set(restored.state_dict()), set(checkpoint))
                self.assertTrue(any(k.startswith("blocks.0.lru.") for k in checkpoint))
                u = torch.randn(2, 7, 2)
                with torch.no_grad():
                    torch.testing.assert_close(restored(u, mode="scan")[0], model(u, mode="scan")[0])

    def test_retained_expert_prototypes_return_streaming_state_in_both_modes(self):
        from neural_ssm.ssm.experimental import (
            ExpertSelectiveTimeVaryingSSM, Block2x2SelectiveBCDExpertsL2SSM,
        )
        u = torch.randn(2, 7, 2)
        cells = (
            ExpertSelectiveTimeVaryingSSM(4, 2, 2, n_experts=2),
            Block2x2SelectiveBCDExpertsL2SSM(4, 2, 2, n_experts=2),
        )
        for cell in cells:
            with self.subTest(cell=type(cell).__name__), torch.no_grad():
                loop = cell(u, mode="loop", return_state=False)
                scan = cell(u, mode="scan", return_state=False)
                self.assertEqual(loop[1].shape, (2, 4))
                torch.testing.assert_close(loop[0], scan[0])
                torch.testing.assert_close(loop[1], scan[1])
                left, state = cell(u[:, :3], mode="loop", return_state=False)
                right, last = cell(u[:, 3:], state=state, mode="scan", return_state=False)
                torch.testing.assert_close(torch.cat((left, right), dim=1), scan[0])
                torch.testing.assert_close(last, scan[1])

    @unittest.skipUnless(importlib.util.find_spec("nonlinear_benchmarks"), "optional benchmark dependency unavailable")
    def test_benchmark_and_ui_choices_match_and_build_current_and_legacy_cells(self):
        from Test_files import benchmark_ui as ui
        from Test_files import run_benchmarks as runner
        from Test_files.Benchmark import ModelConfig, build_model_from_config
        self.assertEqual(set(ui.MODELS), set(runner.MODEL_ZOO))
        self.assertEqual(ui.LEGACY_MODELS, set(runner.LEGACY_MODEL_ZOO))
        self.assertEqual(set(runner.LEGACY_MODEL_ZOO), set(LEGACY_PARAMETRIZATIONS))
        self.assertEqual(
            {n for n, s in runner.MODEL_ZOO.items() if s["kind"] == "deepssm"},
            set(SSM_PARAMETRIZATIONS),
        )
        cfg = runner.GlobalModelConfig(d_model=4, d_state=4, n_layers=1)
        u = torch.randn(2, 7, 2)
        for name in SSM_PARAMETRIZATIONS:
            with self.subTest(param=name), torch.no_grad():
                model = runner.build_model(name, 2, 2, cfg).eval()
                self.assertEqual(model(u).shape, u.shape)
        default_model = build_model_from_config(ModelConfig(n_u=2, n_y=2))
        self.assertEqual(default_model.config.param, "defect")

    @unittest.skipUnless(importlib.util.find_spec("nonlinear_benchmarks"), "optional benchmark dependency unavailable")
    def test_ui_legacy_labels_map_to_cli_keys_and_forward_initialization(self):
        from Test_files import benchmark_ui as ui
        from Test_files import run_benchmarks as runner

        # Exercise command generation without creating a desktop window.
        values = dict(
            epochs="1", batch_size="2", lr="0.001", plot_every="1",
            plot_max_length="32", d_model="4", d_state="4", n_layers="1",
            d_hidden="4", nl_layers="1", gamma="1", report_metric="rmse",
            param_budget="none", device="cpu", ff="auto", mode="auto",
            init="rand", l2ru_eye_scale="0.01", l2ru_rand_scale="0.01",
            defect_block_size="2", defect_rho="0.9", defect_max_radius="0.999",
            defect_factor_margin="0.001", defect_init_scale="0.02",
            defect_state_metric="full",
            defect_max_phase="0.2", defect_phase_center="0",
        )

        def variable(value):
            return SimpleNamespace(get=lambda: value)

        panel = ui.BenchmarkUI.__new__(ui.BenchmarkUI)
        panel.vars = {name: variable(value) for name, value in values.items()}
        panel.bench_list = SimpleNamespace(curselection=lambda: (0,))
        selected = ("defect", "l2ru", "zak", "l2nt")
        panel.model_list = SimpleNamespace(
            curselection=lambda: tuple(ui.MODELS.index(name) for name in selected),
        )
        panel.out_var = variable("/tmp/ssm-package-ui-test")
        for name in ("show_var", "gif_var", "amp_var", "compile_var", "pcg_var",
                     "cudagraph_var", "defect_random_phase_var"):
            setattr(panel, name, variable(False))
        command = panel._build_command()
        args = runner.build_parser().parse_args(command[3:])
        self.assertEqual(args.models, list(selected))
        self.assertEqual(args.init, "rand")
        self.assertEqual(command.count("--init"), 1)
        self.assertEqual(args.defect_block_size, 2)
        self.assertEqual(args.defect_state_metric, "full")
        self.assertFalse(args.defect_random_phase)


if __name__ == "__main__":
    unittest.main()
