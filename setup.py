import setuptools
from cvinfer.version import __version__


setuptools.setup(
    name="cvinfer",
    version=__version__,
    author="Dat Tran",
    author_email="viebboy@gmail.com",
    description="Computer Vision Inference Toolkit",
    long_description="Computer Vision Inference Toolkit",
    long_description_content_type="text",
    license="LICENSE.txt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Operating System :: POSIX",
    ],
)
