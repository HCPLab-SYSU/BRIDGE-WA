from pathlib import Path
from setuptools import find_packages, setup


long_description = (Path(__file__).parent / "README.md").read_text()

core_requirements = [
    "gym",
    "ipdb",
    "mujoco",
    "dm_control",
    "imageio",
]

bridge_wa_eval_requirements = [
    "numpy==1.25.0",
    "scipy==1.14.0",
    "mujoco==3.2.2",
    "dm-control==1.0.22",
    "gym==0.26.2",
    "mediapy==1.2.0",
    "open3d==0.18.0",
    "opencv-python==4.10.0.84",
    "h5py==3.11.0",
    "colorama==0.4.6",
    "colorlog==6.9.0",
    "gdown>=5,<7",
    "imageio>=2.34,<3",
    "ipdb>=0.13,<1",
    "json-numpy==2.1.0",
    "networkx>=3.3,<4",
    "openai>=1.0",
    "pillow>=10,<12",
    "PyYAML>=6,<7",
    "requests>=2.31,<3",
    "scikit-learn>=1.5,<2",
    "tqdm>=4.66,<5",
]

setup(
    name="VLABench",
    version="0.1",
    author="Shiduo Zhang",
    url="",
    description="A large-scale benchmark for language-instruction manipulation tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">3.8",
    install_requires=core_requirements,
    extras_require={"bridge-wa-eval": bridge_wa_eval_requirements},
)
