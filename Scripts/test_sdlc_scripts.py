#!/usr/bin/env python3
"""Unit tests for Scripts/lint_skills.py and Scripts/log_papercut.py."""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_basis  # noqa: E402
import lint_package_dag  # noqa: E402
import lint_skills  # noqa: E402
import log_papercut  # noqa: E402

VALID_SKILL = """---
name: sample-skill
description: "Use when verifying sample behavior."
---

# Sample Skill

1. Run `python3 Scripts/lint_skills.py` to check hygiene.

## Error Recovery

| Failure | Fix |
| :--- | :--- |
| Lint error | Update frontmatter and re-check with `python3 Scripts/lint_skills.py` |
"""


class SkillLinterTests(unittest.TestCase):
    def _make_repo(self, root: Path, skill_text: str, name: str = "sample-skill") -> Path:
        (root / "Scripts").mkdir(parents=True, exist_ok=True)
        (root / "Scripts" / "lint_skills.py").write_text("# ok\n", encoding="utf-8")
        sdir = root / ".agents" / "skills" / name
        sdir.mkdir(parents=True, exist_ok=True)
        spath = sdir / "SKILL.md"
        spath.write_text(skill_text, encoding="utf-8")
        return spath

    def _run_lint(self, root: Path) -> tuple[int, str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = lint_skills.main(["--repo-root", str(root), str(root / ".agents" / "skills")])
        return rc, buf.getvalue()

    def test_valid_skill_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_repo(root, VALID_SKILL)
            rc, out = self._run_lint(root)
            self.assertEqual(rc, 0, out)
            self.assertIn("0 violation(s)", out)

    def test_s1_missing_frontmatter_and_name_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_repo(root, "# No FM\n\n1. Run `make lint`\n")
            rc, out = self._run_lint(root)
            self.assertEqual(rc, 1)
            self.assertIn("S1", out)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = VALID_SKILL.replace("name: sample-skill", "name: wrong-name")
            self._make_repo(root, bad)
            rc, out = self._run_lint(root)
            self.assertEqual(rc, 1)
            self.assertIn("S1 name 'wrong-name'", out)

    def test_s1_overlong_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            long_desc = "x" * 505
            bad = VALID_SKILL.replace("Use when verifying sample behavior.", long_desc)
            self._make_repo(root, bad)
            rc, out = self._run_lint(root)
            self.assertEqual(rc, 1)
            self.assertIn("S1 description", out)

    def test_s2_max_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = VALID_SKILL + ("\nline\n" * 130)
            self._make_repo(root, bad)
            rc, out = self._run_lint(root)
            self.assertEqual(rc, 1)
            self.assertIn("S2", out)

    def test_s3_missing_steps_or_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            no_cmd = VALID_SKILL.replace("`python3 Scripts/lint_skills.py`", "the linter")
            self._make_repo(root, no_cmd)
            rc, out = self._run_lint(root)
            self.assertEqual(rc, 1)
            self.assertIn("S3", out)

    def test_s4_missing_or_weak_error_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weak = VALID_SKILL.replace("Update frontmatter and re-check with `python3 Scripts/lint_skills.py`", "Try again")
            self._make_repo(root, weak)
            rc, out = self._run_lint(root)
            self.assertEqual(rc, 1)
            self.assertIn("S4 weak recovery", out)

    def test_s5_broken_link_and_missing_repo_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = VALID_SKILL + "\nSee [missing](./references/nope.md) and `Scripts/missing.py`.\n"
            self._make_repo(root, bad)
            rc, out = self._run_lint(root)
            self.assertEqual(rc, 1)
            self.assertIn("S5 broken relative link", out)
            self.assertIn("S5 missing repo path", out)

    def test_s6_forbids_mcp_json_and_s7_forbids_jinja(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spath = self._make_repo(root, VALID_SKILL + "\n`{{ chat_template }}`\n")
            (spath.parent / ".mcp.json").write_text("{}", encoding="utf-8")
            rc, out = self._run_lint(root)
            self.assertEqual(rc, 1)
            self.assertIn("S6", out)
            self.assertIn("S7", out)

    def test_s8_forbids_external_naming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_tok = "la" + "ya_guard"
            self._make_repo(root, VALID_SKILL + f"\nCall `{legacy_tok}` first.\n")
            rc, out = self._run_lint(root)
            self.assertEqual(rc, 1)
            self.assertIn("S8", out)


class PapercutTests(unittest.TestCase):
    def test_triage_empty_and_log_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pfile = Path(tmp) / "papercuts.jsonl"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = log_papercut.main(["triage", "--file", str(pfile)])
            self.assertEqual(rc, 0)
            self.assertIn("0 recorded", buf.getvalue())

            for idx in range(2):
                with redirect_stdout(io.StringIO()):
                    rc = log_papercut.main(
                        [
                            "log",
                            "--area",
                            "cli",
                            "--target",
                            "qwg run",
                            "--tried",
                            f"--max-new-tokens {idx}",
                            "--expected",
                            "cap output",
                            "--actual",
                            "unknown flag",
                            "--fix",
                            "use --max-tokens",
                            "--file",
                            str(pfile),
                        ]
                    )
                self.assertEqual(rc, 0)
            lines = pfile.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            rec = json.loads(lines[0])
            self.assertEqual(rec["target"], "qwg run")

            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                rc = log_papercut.main(["triage", "--file", str(pfile), "--limit", "5"])
            self.assertEqual(rc, 0)
            self.assertIn("2 recorded", buf2.getvalue())
            self.assertIn("2x  [cli] qwg run", buf2.getvalue())


class CICacheAndDAGResilienceTests(unittest.TestCase):
    def test_compound_test_dependency_condition_parsed_for_all_branches(self) -> None:
        sample = '''
        let components: [(String, String)] = [
            ("QWRope", "QWRopeTests"), ("QWAttn", "QWAttnTests"), ("QWGDN", "QWGDNTests"),
        ]
        var targets: [Target] = [T("QWCore"), T("QWQuant", ["QWCore"])]
        for (mod, test) in components {
            var deps: [Target.Dependency] = [.byName(name: mod)]
            if mod == "QWRope" || mod == "QWAttn" || mod == "QWGDN" { deps.append(.byName(name: "QWQuant")) }
        }
        '''
        _, tests = lint_package_dag.parse_package_swift(sample)
        self.assertIn("QWQuant", tests["QWRopeTests"])
        self.assertIn("QWQuant", tests["QWAttnTests"])
        self.assertIn("QWQuant", tests["QWGDNTests"])

    def test_basis_cache_key_invalidates_on_any_metal_or_header_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mroot = root / "QWMetal"
            (mroot / "include").mkdir(parents=True)
            mtl = mroot / "sample.metal"
            hdr = mroot / "include" / "lut.metalh"
            script = root / "check_basis.py"
            mtl.write_text("kernel void k() {}\n", encoding="utf-8")
            hdr.write_text("constant float c = 1.0f;\n", encoding="utf-8")
            script.write_text("# v1\n", encoding="utf-8")
            cc = ["/usr/bin/true"]

            k1 = check_basis.compute_basis_cache_key(mroot, cc, script_path=script)
            # Touching an ignored .metallib output must NOT invalidate the source key.
            (mroot / "kernel.metallib").write_bytes(b"\x00\x01")
            self.assertEqual(k1, check_basis.compute_basis_cache_key(mroot, cc, script_path=script))
            # 1-byte header edit MUST invalidate immediately.
            hdr.write_text("constant float c = 2.0f;\n", encoding="utf-8")
            k2 = check_basis.compute_basis_cache_key(mroot, cc, script_path=script)
            self.assertNotEqual(k1, k2)
            # Script flag edit MUST invalidate immediately.
            script.write_text("# v2\n", encoding="utf-8")
            k3 = check_basis.compute_basis_cache_key(mroot, cc, script_path=script)
            self.assertNotEqual(k2, k3)


if __name__ == "__main__":
    unittest.main()
