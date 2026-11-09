import unittest
from unittest.mock import patch

import gen_kernel_inventory
import lint_kernels


class KernelInventoryTests(unittest.TestCase):
    def test_missing_dispatch_and_missing_metallib_kernel_are_reported(self):
        inventory = gen_kernel_inventory.load_manifest()
        missing = inventory["common"][0]
        modified = {key: list(value) for key, value in inventory.items()}
        modified["common"].remove(missing)
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

    def test_stale_generated_inventory_is_reported(self):
        inventory = gen_kernel_inventory.load_manifest()
        available = set().union(*(set(names) for names in inventory.values()))
        with patch.object(gen_kernel_inventory, "render", return_value="stale"), \
             patch.object(lint_kernels, "dispatched_in", return_value=set()), \
             patch.object(lint_kernels, "chain_kernels", return_value=set()):
            problems = lint_kernels.door_problems(available)
        self.assertTrue(any("STALE KERNEL INVENTORY" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
