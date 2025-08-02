# PhaseT3M

**PhaseT3M** is a Python-based reconstruction package for **High-Resolution Transmission Electron Microscopy (HRTEM)** data. It enables robust 3D imaging via nonlinear phase retrieval and is particularly suited for cryo-ET datasets involving focal series and tilt series.

---

## ✨ Features

1. **Focal Series Reconstruction**  
   Performs precise phase retrieval from through-focal image stacks.

2. **3D Phase Retrieval-based Tomography**  
   Combines multiple tilted focal series to reconstruct high-resolution 3D volumes.

> 🧩 This package follows the structure of **[py4DSTEM](https://github.com/py4dstem/py4DSTEM)**, ensuring compatibility and familiarity for users of similar scientific tools.

---

## 🚀 Installation and Run

### 📦 Installation

Create and activate a new conda environment:

```bash
conda create -n PhaseT3M python=3.11
conda activate PhaseT3M
```

If you need GPU acceleration:
```bash
conda install -c conda-forge cupy
```

Install the PhaseT3M:
```bash
pip install -e .
```

Run the example
```bash
# Focal series phase retrieval
python ./examples/test_focal_recon.ipynb

# 3D reconstruction from tilt series
python ./examples/test_tilt_recon.ipynb
```
