# pylint: disable=missing-docstring
import setuptools

setuptools.setup(
    name="n2c2_ss",
    description="Structured sequence approach to n2c2 2022 track 2 social determinants of health (SDoH) challenge",
    author="Brian Romanowski",
    author_email="romanows@gmail.com",

    install_requires=[
        "transformers",
    ],

    license="secret",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=setuptools.find_packages(),
)
