import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from score_teacher_logits import main, score


class TeacherLogitTests(unittest.TestCase):
    def setUp(self):
        self.prefixes = np.array([[0, -1], [0, 1]], dtype=np.int32)
        self.logits = np.array([[0.0, 1.0], [1000.0, 999.0]], dtype=np.float32)

    def test_identical_full_distributions_have_zero_kl(self):
        result = score(self.logits, self.logits, self.prefixes, self.prefixes)
        self.assertEqual(result["mean_kl"], 0)
        self.assertEqual(result["top1_agreement"], 1)
        self.assertEqual(result["positions"], 2)

    def test_known_distribution_and_nll_delta(self):
        teacher = np.log(np.array([[0.8, 0.2]]))
        candidate = np.log(np.array([[0.5, 0.5]]))
        contexts = np.array([[0]], dtype=np.int32)
        result = score(teacher, candidate, contexts, contexts, np.array([0]))
        expected = 0.8 * np.log(0.8 / 0.5) + 0.2 * np.log(0.2 / 0.5)
        self.assertAlmostEqual(result["mean_kl"], expected)
        self.assertAlmostEqual(result["mean_nll_delta"], np.log(0.8 / 0.5))

    def test_rejects_diverged_contexts_and_nonfinite_logits(self):
        changed = self.prefixes.copy()
        changed[1, 1] = 0
        with self.assertRaisesRegex(ValueError, "identical integer token prefixes"):
            score(self.logits, self.logits, self.prefixes, changed)
        invalid = self.logits.copy()
        invalid[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            score(invalid, self.logits, self.prefixes, self.prefixes)

    def test_rejects_invalid_padding_and_targets(self):
        bad = np.array([[-1, 0], [0, 1]], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "right edge"):
            score(self.logits, self.logits, bad, bad)
        with self.assertRaisesRegex(ValueError, "targets"):
            score(self.logits, self.logits, self.prefixes, self.prefixes, np.array([0, 2]))

    def test_cli_persists_metrics_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logits = root / "logits.npy"
            contexts = root / "contexts.npy"
            output = root / "metrics.json"
            np.save(logits, self.logits)
            np.save(contexts, self.prefixes)
            args = [
                "--teacher-logits", str(logits), "--candidate-logits", str(logits),
                "--teacher-contexts", str(contexts), "--candidate-contexts", str(contexts),
                "--teacher-id", "synthetic-teacher", "--candidate-id", "synthetic-candidate",
                "--output", str(output),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["mean_kl"], 0)
            self.assertEqual(len(result["teacher_sha256"]), 64)
            self.assertEqual(len(result["corpus_sha256"]), 64)
            original = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
