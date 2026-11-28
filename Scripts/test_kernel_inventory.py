import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gen_kernel_inventory
import lint_kernels


class KernelInventoryTests(unittest.TestCase):
    def test_missing_dispatch_and_missing_metallib_kernel_are_reported(self):
        inventory = gen_kernel_inventory.load_manifest()
        for group in ("common", "hyper_only"):
            missing = inventory[group][0]
            modified = {key: list(value) for key, value in inventory.items()}
            modified[group].remove(missing)
            available = set().union(*(set(names) for names in modified.values()))
            with patch.object(gen_kernel_inventory, "load_manifest", return_value=modified), \
                 patch.object(lint_kernels, "dispatched_in", return_value={missing}), \
                 patch.object(lint_kernels, "chain_kernels", return_value=set()):
                problems = lint_kernels.door_problems(available)
            self.assertTrue(any("UNDECLARED DOOR" in problem and missing in problem for problem in problems))

            with patch.object(lint_kernels, "dispatched_in", return_value=set()), \
                 patch.object(lint_kernels, "chain_kernels", return_value=set()):
                problems = lint_kernels.door_problems(available)
            self.assertTrue(any("MISSING INVENTORY KERNEL" in problem and missing in problem for problem in problems))
            self.assertTrue(any("STALE INVENTORY DOOR" in problem and missing in problem for problem in problems))

    def test_duplicate_across_groups_is_rejected(self):
        inventory = gen_kernel_inventory.load_manifest()
        bad = {key: list(value) for key, value in inventory.items()}
        bad["hyper_only"].append(bad["dense_only"][0])
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_manifest = Path(tmpdir) / "kernel_inventory.json"
            bad_manifest.write_text(json.dumps(bad), encoding="utf-8")
            with patch.object(gen_kernel_inventory, "MANIFEST", bad_manifest):
                with self.assertRaises(ValueError):
                    gen_kernel_inventory.load_manifest()

    def test_stale_generated_inventory_is_reported(self):
        inventory = gen_kernel_inventory.load_manifest()
        available = set().union(*(set(names) for names in inventory.values()))
        with patch.object(gen_kernel_inventory, "render", return_value="stale"), \
             patch.object(lint_kernels, "dispatched_in", return_value=set()), \
             patch.object(lint_kernels, "chain_kernels", return_value=set()):
            problems = lint_kernels.door_problems(available)
        self.assertTrue(any("STALE KERNEL INVENTORY" in problem for problem in problems))

    def test_stale_nax_registry_is_reported(self):
        nax_names = gen_kernel_inventory.load_nax_manifest()
        with patch.object(gen_kernel_inventory, "render_nax_registry", return_value="stale"):
            problems = lint_kernels.nax_registry_problems(set(nax_names))
        self.assertTrue(any("STALE NAX REGISTRY" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
