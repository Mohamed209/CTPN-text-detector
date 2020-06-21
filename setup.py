import pathlib
from setuptools import setup, find_packages
import setuptools


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
    version="2.0.1",
    install_requires=load_req(),
    include_package_data=True,
    description="encapsulating CTPN text detector in python package",
    long_description=README,
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
