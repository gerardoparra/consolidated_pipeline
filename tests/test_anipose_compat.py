import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


COMPAT_DIR = (
    Path(__file__).resolve().parent.parent / "pipeline" / "_anipose_compat"
)


class AniposeCompatibilityTests(unittest.TestCase):
    def test_filter_subprocess_accepts_legacy_positional_hdf_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "filtered.h5"
            env = os.environ.copy()
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = str(COMPAT_DIR) + (
                os.pathsep + existing if existing else ""
            )
            code = (
                "import pandas as pd, sys; "
                "pd.DataFrame({'x': [1.0]}).to_hdf("
                "sys.argv[1], 'df_with_missing', format='table', mode='w')"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code, str(output)],
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(pd.read_hdf(output).loc[0, "x"], 1.0)


if __name__ == "__main__":
    unittest.main()
