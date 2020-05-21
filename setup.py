import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fmdtools",
    version="0.5.4",
    author="Daniel Hulse",
    author_email="hulsed@oregonstate.edu",
    description="System modelling and fault-injection-based resilience assessment in Python ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DesignEngrLab/fmdtools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)