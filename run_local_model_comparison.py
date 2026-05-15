"""Legacy entry point for the Phi-2-only experiment.

This wrapper exists so old shell history that calls this file still runs the
current Phi-2 pipeline.
"""
from __future__ import annotations

from run_experiment import main


if __name__ == "__main__":
    main()
