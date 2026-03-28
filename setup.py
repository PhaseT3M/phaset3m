from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup
from distutils.util import convert_path

with open("README.md", "r") as f:
    long_description = f.read()

version_ns = {}
vpath = convert_path("PhaseT3M/version.py")
with open(vpath) as version_file:
    exec(version_file.read(), version_ns)

setup(
    name="PhaseT3M",
    version=version_ns["__version__"],
    packages=find_packages(),
    description="An open source 3D reconstruction package using multiple tilted HRTEM focal series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Juhyeok Lee",
    author_email="jhlee0667@lbl.gov",
    keywords="HRTEM 3D reconstruciton",
    python_requires=">=3.10",
    install_requires=[
        "numpy >= 2.4",
        "scipy >= 1.7.7",
        "h5py >= 3.15.0",
        "hdf5plugin >= 6.0.0",
        "matplotlib >= 3.10.0",
        "scikit-image >= 0.26.0",
        "scikit-learn >= 1.8.0",
        "scikit-optimize >= 0.10.0",
        "tqdm >= 4.46.3",
        "mrcfile >= 1.5.4",
        "tifffile >= 1.3.0",
        # "dask >= 2.3.0",
        # "distributed >= 2.3.0",
        # "mpire >= 2.7.1",
        # "threadpoolctl >= 3.1.0",
        # "pylops >= 2.1.0",
    ],
    extras_require={
        "ipyparallel": ["ipyparallel >= 6.2.4", "dill >= 0.3.3"],
        "cuda": ["cupy >= 12.0.0"],
    },
)
