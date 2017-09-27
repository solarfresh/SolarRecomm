from setuptools import setup, find_packages

setup(
    name='solarrecomm',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    description='recommender system based on deep learning',
    install_requires=[
        'tensorflow>=1.3.0',
    ],
)