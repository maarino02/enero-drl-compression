# Dataset location: change this path to where you unzipped the ENERO dataset
DATASETS_DIR = "path/to/Enero_datasets"

# -----------------------------------------------------------------------------
# Everything below is derived from the line above; there is no need to edit it.
# -----------------------------------------------------------------------------
import os as _os

# The ENERO_DATASETS environment variable, if set, takes precedence over DATASETS_DIR.
# Paths use forward slashes because the environments split them on "/TRAIN".
DATASETS_DIR = _os.environ.get("ENERO_DATASETS", DATASETS_DIR).replace("\\", "/").rstrip("/")

# Root of the code/ folder: all default paths are built from here, so the scripts
# do not depend on the directory they are launched from.
CODE_DIR = _os.path.dirname(_os.path.abspath(__file__)).replace("\\", "/")
MODELS_DIR = CODE_DIR + "/models"
RESULTS_DIR = CODE_DIR + "/results"
LOGS_DIR = CODE_DIR + "/logs"
FIGURES_OUTPUT_DIR = CODE_DIR + "/figures/output"

# Training and main evaluation topologies: NEW_<Topology>/{TRAIN,EVALUATE}
MAIN_TOPOLOGIES_DIR = DATASETS_DIR + "/dataset_sing_top/data/results_my_3_tops_unif_05-1"
# Topology Zoo topologies used in the scalability study: <Topology>/
ZOO_TOPOLOGIES_DIR = DATASETS_DIR + "/results-1-link_capacity-unif-05-1/results_zoo"
