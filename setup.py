from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="DARAZ PRODUCT RECOMMENDER",
    version="0.1",
    author="Anas",
    packages=find_packages(),
    install_requires = requirements,
)
