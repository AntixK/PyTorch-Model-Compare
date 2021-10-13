import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch_cka",
    version="0.1",
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
    package_dir={"": "torch_cka"},
    packages=setuptools.find_packages(where="torch_cka"),
    python_requires=">=3.7",
)