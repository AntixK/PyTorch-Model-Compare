import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch_cka",
    version="0.21",
    author="Anand K Subramanian",
    author_email="anandkrish894@gmail.com",
    description="A package to compare neural networks by their feature similarity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AntixK/PyTorch-Model-Compare",
    project_urls={
        "Bug Tracker": "https://github.com/AntixK/PyTorch-Model-Compare/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)