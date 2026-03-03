from __future__ import annotations

import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from robot_model.robot_model_vis import main_surface_graph_cli


if __name__ == "__main__":
    main_surface_graph_cli()
