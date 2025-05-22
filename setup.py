
from setuptools import setup, find_packages
from os import path
import re

package_name="onnx2webnn"
root_dir = path.abspath(path.dirname(__file__))

with open("README.md") as f:
    long_description = f.read()

with open(path.join(root_dir, package_name, '__init__.py')) as f:
    init_text = f.read()
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)

setup(
    name=package_name,
    version=version,
    description=\
        "Exports the ONNX file to a WebNN JavaScript file and a bin file containing the weights.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ningxin Hu",
    author_email="ningxin.hu@intel.com",
    url="https://github.com/huningxin/onnx2webnn",
    license="MIT License",
    packages=find_packages(),
    platforms=["linux", "unix"],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            "onnx2webnn=onnx2webnn:main"
        ]
    }
)
