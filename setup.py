import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="snax",
    version="0.0.1",
    author="Dieterich Lawson",
    author_email="dieterich.lawson@gmail.com",
    description="A simple functional deep learning library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dieterichlawson",
    packages=["snax"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'jax', 'jaxlib', 'tfp-nightly'
      ],
    python_requires='>=3.6',
)
