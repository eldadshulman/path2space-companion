from setuptools import find_packages, setup

setup(
    name="path2space",
    version="0.1.0",
    description="H&E -> spatial gene-expression prediction (path2space, companion to the paper).",
    packages=find_packages(exclude=["tests", "scripts", "examples"]),
    python_requires=">=3.10",
    install_requires=[
        # Hard pins live in environment.yml; this list is best-effort for
        # pip-only installs and intentionally loose.
        "numpy",
        "pandas",
        "scipy",
        "scikit-image",
        "opencv-python",
        "openslide-python",
        "Pillow",
        "torch",
        "torchvision",
        "timm",
        "spams-bin",
    ],
    include_package_data=True,
)
