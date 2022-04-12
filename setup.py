import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

required = ["scipy","tqdm","networkx","numpy","matplotlib","pandas","ordered-set","dill"]

setuptools.setup(
    name="fmdtools",
    version="1.1.0",
    author="Daniel Hulse",
    author_email="daniel.e.hulse@nasa.gov",
    description="System resilience modelling, simulation, and assessment in Python",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/nasa/fmdtools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved ",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=required,
)
