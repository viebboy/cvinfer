import setuptools
from construct_image_blob.version import __version__


setuptools.setup(
    name="construct_image_blob",
    version=__version__,
    author="Dat Tran",
    author_email="hello@dats.bio",
    description="construct binary blobs from image directory",
    long_description="construct binary blobs from image directory",
    long_description_content_type="text",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Operating System :: POSIX",
    ],
    entry_points={
        "console_scripts": [
            "construct-image-blob = construct_image_blob.cli:main",
        ]
    },
)
