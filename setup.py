import pathlib
from setuptools import setup, find_packages
from setuptools.command.install import install
import setuptools
import subprocess
import atexit
import distutils


def load_req():
    with open('requirements.txt') as req:
        return [str(l) for l in req.readlines()]


def build_cython_modules(build_path='src/utils/bbox/'):
    subprocess.run(['pip3', 'install', 'requests'])
    import requests
    import os
    files = ['nms.pyx', 'bbox.pyx', 'make.sh']
    os.chdir(build_path)
    for file in files:
        res = requests.get(
            url='https://raw.githubusercontent.com/Mohamed209/CTPN-text-detector/master/src/utils/bbox/'+file)
    with open(file, mode='w') as code:
        code.writelines(res.text)
    subprocess.run(['chmod', '+x', 'make.sh'])
    subprocess.run(['bash', 'make.sh'])


class my_install(install):
    def run(self):
        install.run(self)
        self.execute(build_cython_modules, self.install_lib,
                     msg='Building Cython Modules :)')


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
try:
    setup(
        name="ctpn-text-detector",
        version="2.0.7",
        install_requires=load_req(),
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
finally:
    build_cython_modules()
