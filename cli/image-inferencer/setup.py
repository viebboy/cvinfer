import setuptools
from image_inferencer.version import __version__


setuptools.setup(
    name="image_inferencer",
    version=__version__,
    author="Dat Tran",
    author_email="hello@dats.bio",
    description="cli to perform image inference using multiple workers",
    long_description="cli to perform image inference using multiple workers",
    long_description_content_type="text",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Operating System :: POSIX",
    ],
    entry_points={
        "console_scripts": [
            "image-inferencer = image_inferencer.cli:main",
        ]
    },
)
