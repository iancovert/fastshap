import setuptools

setuptools.setup(
    name="fastshap-torch",
    version="0.0.1",
    author="Ian Covert",
    author_email="icovert@cs.washington.edu",
    description="An amortized approach for calculating local Shapley value explanations.",
    long_description="""
        FastSHAP is an amortized approach for calculating Shapley value
        explanations for many examples. It involves learning a model that
        outputs Shapley value estimates in a single forward pass.
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/iancovert/fastshap/",
    packages=['fastshap'],
    install_requires=[
        'numpy',
        'torch',
        'tqdm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
)
