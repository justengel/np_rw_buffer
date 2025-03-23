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

logging.basicConfig(
    level=logging.INFO,  # Show INFO and above
    format="%(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler()  # Output to console (stderr)
    ]
)
log = logging.getLogger("np_rw_buffer.setup")

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
                raise BuildFailed(f"Failed to run build_ext: {x}")

        def build_extension(self, ext):
            try:
                build_ext.build_extension(self, ext)
            except ext_errors as x:
                raise BuildFailed(f"Failed to build extension {ext.name}: {x}")
    return WrappedBuildExt


def get_extensions():
    """Dynamically define extensions to handle optional NumPy import."""
    try:
        import numpy
        log.info("NumPy found, attempting to build C extension.")
        return [
            Extension(
                'np_rw_buffer._circular_indexes',
                # define_macros=[('MAJOR_VERSION', '1')],
                # extra_compile_args=['-std=c99'],
                sources=['src/np_rw_buffer/_circular_indexes.c'],
                include_dirs=['src/np_rw_buffer', numpy.get_include()],
                extra_compile_args=["-O2"],  # Optional: compiler flags
            )
        ]
    except ImportError:
        log.warning("NumPy not found; skipping C extension.")
        return []


if __name__ == "__main__":
    # Variables
    meta = get_meta('src/np_rw_buffer/__meta__.py')
    name = meta['name']
    version = meta['version']
    description = meta['description']
    url = meta['url']
    author = meta['author']
    author_email = meta['author_email']
    keywords = 'read write ring circular buffer'
    packages = find_packages(where="src")

    setup_kwargs = {
        'name': name,
        'version': version,
        'description': description,
        'long_description': read('README.rst'),
        'keywords': keywords,
        'url': url,
        'download_url': ''.join((url, '/archive/v', version, '.tar.gz')),
        'author': author,
        'author_email': author_email,
        'platform': 'any',
        'packages': packages,
        'package_dir': {'': 'src'},  # Map root to src/
        'include_package_data': True,
        'package_data': {
            'np_rw_buffer': ['*.pyi', '*.c']
        },
        'install_requires': ['numpy>=1.26.0'],
        'extras_require': {},
    }

    # Comment this out if you want the simple setup
    cmd_classes = setup_kwargs.setdefault('cmdclass', {})
    setup_kwargs['cmdclass']['build_ext'] = construct_build_ext(build_ext)

    # Get extensions dynamically
    extensions = get_extensions()

    if extensions:
        try:
            # Run the setup with the c code
            setup(ext_modules=extensions, **setup_kwargs)
            log.info("Successfully built with C extension.")
        except BuildFailed as err:
            log.warning(f"C extension build failed: {err}")

            # Fall back to pure Python
            if 'build_ext' in setup_kwargs['cmdclass']:
                del setup_kwargs['cmdclass']['build_ext']

            # Run the setup without the c code
            setup(**setup_kwargs)
            log.info("Plain-Python installation succeeded.")
    else:
        # No extensions available (e.g., NumPy missing), proceed with pure Python
        setup(**setup_kwargs)
        log.info("Plain-Python installation succeeded (no C extension attempted).")