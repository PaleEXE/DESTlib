from setuptools import setup, find_packages

setup(
        name='destlib',
        version='0.0.1',
        packages=find_packages(),
        install_requires=[
            'matplotlib~=3.10.0',
            'numpy~=2.2.1',
            'pandas~=2.2.3',
        ]
)