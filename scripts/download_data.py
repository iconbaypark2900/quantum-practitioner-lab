"""Pre-fetch the source datasets the tutorials need.

Currently just Hetionet (~12 MB), used by the quantum-kernel tutorial. Files are
cached under ``data/raw`` (override with ``QPRAC_DATA_DIR``) and are gitignored.
"""

from __future__ import annotations

import sys

from qprac_lab.data.hetionet import default_cache_dir, download_hetionet, hetionet_available


def main() -> int:
    cache = default_cache_dir()
    if hetionet_available():
        print(f"Hetionet already cached at {cache}")
        return 0
    print(f"Downloading Hetionet (~12 MB) to {cache} ...", flush=True)
    try:
        path = download_hetionet()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
