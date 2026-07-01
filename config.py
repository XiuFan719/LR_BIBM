"""Dataset paths. No data ships with this repository.

Obtain the three public datasets from their official sources and set DATA_ROOT
(environment variable) to the directory that contains them, or edit the defaults
below.

  HCP S1200  : per-subject ROI time-series .mat (Brainnetome, 246 ROIs) + phenotype csv
  ABIDE      : <root>/<Healthy|ASD>/*_rois_aal.1D (AAL, 116 ROIs)
  ADHD-200   : <root>/<0|1>/*session_1_rest_1*aal_TCs.1D (AAL, 116 ROIs) + phenotypic tsv
"""
import os

DATA_ROOT = os.environ.get("DATA_ROOT", "./data")

# HCP: any directory tree of per-subject .mat files; the subject id is the
# leading token of the file name and is matched against the phenotype table.
HCP_ROOT = os.path.join(DATA_ROOT, "hcp")
HCP_INFO_CSV = os.path.join(HCP_ROOT, "information.csv")

ABIDE_ROOT = os.path.join(DATA_ROOT, "abide")

ADHD_ROOT = os.path.join(DATA_ROOT, "adhd")
ADHD_PHENO = os.path.join(DATA_ROOT, "adhd", "adhd200_preprocessed_phenotypics.tsv")
