"""
setup.py - Setup file to distribute the library

See Also:
    https://github.com/pypa/sampleproject
    https://packaging.python.org/en/latest/distributing.html
    https://pythonhosted.org/an_example_pypi_project/setuptools.html
"""
import os
import glob
from setuptools import setup, Extension, find_packages

import numpy


def read(fname):
    """Read in a file"""
    with open(os.path.join(os.path.dirname(__file__), fname), 'r') as file:
        return file.read()


def get_meta(filename):
    """Return the metadata dictionary from the given filename."""
    with open(filename, 'r') as f:
        meta = {}
        exec(compile(f.read(), filename, 'exec'), meta)
        return meta


# ========== Optional C extension ==========
import logging
from setuptools.command.build_ext import build_ext
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError

logging.basicConfig()
log = logging.getLogger(__file__)

ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError)


class BuildFailed(Exception):
    pass


def construct_build_ext(build_ext):
    class WrappedBuildExt(build_ext):
        # This class allows C extension building to fail.
        def run(self):
            try:
                build_ext.run(self)
            except DistutilsPlatformError as x:
                raise BuildFailed(x)

        def build_extension(self, ext):
            try:
                build_ext.build_extension(self, ext)
            except ext_errors as x:
                raise BuildFailed(x)
    return WrappedBuildExt


if __name__ == "__main__":
    # Variables
    meta = get_meta('np_rw_buffer/__meta__.py')
    name = meta['name']
    version = meta['version']
    description = meta['description']
    url = meta['url']
    author = meta['author']
    author_email = meta['author_email']
    keywords = 'read write ring circular buffer'
    packages = find_packages(exclude=('tests', 'bin'))

    # Extensions
    extensions = [
        Extension('np_rw_buffer._circular_indexes',
                  # define_macros=[('MAJOR_VERSION', '1')],
                  # extra_compile_args=['-std=c99'],
                  sources=['src/circular_indexes.c'],
                  include_dirs=['src', numpy.get_include()]),
        ]
    setup_kwargs = {'name': name,
                    'version': version,
                    'description': description,
                    'long_description': read('README.rst'),
                    'keywords': keywords,
                    'url': url,
                    'download_url': ''.join((url, '/archive/v', version, '.tar.gz')),

                    'author': author,
                    'author_email': author_email,

                    'license': 'Proprietary',
                    'platform': 'any',
                    'classifiers': ['Programming Language :: Python',
                                    'Programming Language :: Python :: 3',
                                    'Operating System :: OS Independent'],

                    'scripts': [file for file in glob.iglob('bin/*.py')],  # Run with python -m Scripts.module args

                    # 'ext_modules': extensions,
                    'packages': packages,
                    'include_package_data': True,
                    'package_data': {pkg: ['*', '*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*']
                                     for pkg in packages if '/' not in pkg and '\\' not in pkg},

                    'install_requires': [
                        'numpy>=1.13.0',
                        ],
                    'extras_require': {
                        },
                    }

    # Comment this out if you want the simple setup
    cmd_classes = setup_kwargs.setdefault('cmdclass', {})
    setup_kwargs['cmdclass']['build_ext'] = construct_build_ext(build_ext)

    try:
        # Run the setup with the c code
        setup(ext_modules=extensions, **setup_kwargs)
    except BuildFailed as err:
        log.warning(err)
        log.warning("The C extension could not be compiled")

        # Remove any previously defined build_ext command class.
        if 'build_ext' in setup_kwargs['cmdclass']:
            del setup_kwargs['cmdclass']['build_ext']
        if 'build_ext' in cmd_classes:
            del cmd_classes['build_ext']

        # Run the setup without the c code
        setup(**setup_kwargs)
        log.info("Plain-Python installation succeeded.")
