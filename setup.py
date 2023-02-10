import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="snax",
    version="0.0.23",
    author="Dieterich Lawson",
    author_email="dieterich.lawson@gmail.com",
    description="A simple deep learning library for JAX.",
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
          'jax', 'jaxlib', 'tensorflow_probability', 'chex', 'typing_extensions', 'dill',
          'equinox', 'optax', 'wandb'
      ],
    python_requires='>=3.7',
)
