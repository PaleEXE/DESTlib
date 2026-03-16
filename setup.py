from setuptools import setup, find_packages

setup(
    name="destlib",
    version="1.2.0",
    description="A Python library for Discrete Event and Stochastic Simulation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Mohammad Al-Hennawi",
    author_email="mohammed.alhennawi@gmail.com",
    url="https://github.com/PaleEXE/DESTlib",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
