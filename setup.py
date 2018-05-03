"""
    setup.py - Setup file to distribute the library
See Also:
    https://github.com/pypa/sampleproject
    https://packaging.python.org/en/latest/distributing.html
    https://pythonhosted.org/an_example_pypi_project/setuptools.html
"""
import os
import glob

from setuptools import setup

import uuid
from pip.req import parse_requirements


def read(fname):
    """Read in a file"""
    with open(os.path.join(os.path.dirname(__file__), fname), "r") as file:
        return file.read()


if __name__ == "__main__":
    # ===== Requirements =====
    try:
        requirements_file = "requirements.txt"
        requirements = [str(ir.req)
                        for ir in parse_requirements(requirements_file, session=uuid.uuid1())
                        if ir.req is not None]
    except FileNotFoundError:
        requirements = []
    # ===== END Requirements =====

    setup(
        name="np_rw_buffer",
        version="1.0.0",
        description="Library to help store audio data in a circular buffer.",
        url="https://github.com/HashSplat/np_rw_buffer",
        download_url="https://github.com/HashSplat/np_rw_buffer/archive/v1.0.0.tar.gz",

        keywords=["buffer", "read write buffer", "ring buffer", 'ring', "circular buffer", "circular"],

        author="Justin Engel",
        author_email="jtengel08@gmail.com",

        license="MIT",

        platforms="any",
        classifiers=["Programming Language :: Python",
                     "Programming Language :: Python :: 3",
                     "Operating System :: OS Independent"],

        scripts=[],

        long_description=read("README.md"),
        packages=["np_rw_buffer"],
        install_requires=requirements,

        include_package_data=False,

        # package_data={
        #     'package': ['file.dat']
        # }

        # options to install extra requirements
        # extras_require={
        #     'dev': [],
        #     'test': ['converage'],
        # }

        # Data files outside of packages
        # data_files=[('my_data', ["data/my_data.dat"])],

        # keywords='sample setuptools development'

        # entry_points={
        #     'console_scripts': [
        #         'foo = my_package.some_module:main_func',
        #         'bar = other_module:some_func',
        #     ],
        #     'gui_scripts': [
        #         'baz = my_package_gui:start_func',
        #     ]
        # }
    )
