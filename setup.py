import pathlib
import subprocess
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np


def load_req():
    with open('requirements.txt') as req:
        return [str(l) for l in req.readlines()]


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="ctpn-text-detector",
    version="3.0.0",
    install_requires=load_req(),
    ext_modules=cythonize(["src/*.pyx"],
                          include_path=np.get_include()),
    include_package_data=True,
    description="encapsulating CTPN text detector in python package",
    long_description=README,
    long_description_content_type='text/markdown',
    url="https://github.com/Mohamed209/CTPN-text-detector",
    author="Mohamed Mossad",
    author_email="mohamedmosad209@gmail.com",
    license="MIT",
    classifiers=[
            "License :: OSI Approved :: MIT License",
            'Programming Language :: Python :: 3',
    ],
    packages=find_packages()
)
