import setuptools
import os

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(FILE_PATH, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements_path = os.path.join(FILE_PATH, "requirements/base.txt")
with open(requirements_path, encoding="utf-8") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="pyterrain",
    version="0.0.1",
    author="Wentao Li",
    author_email="",
    description="A Python package to fetch terrain data easily.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Clarmy/pyterrain",
    include_package_data=True,
    package_data={"": []},
    packages=setuptools.find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)
