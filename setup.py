from setuptools import setup, find_packages

setup(
    name="rag-meter",
    version="0.0.1",
    author="Marwan Elghitany",
    author_email="marwan.elghitany@cntxt.tech",
    description="Reserved package for ragmeter",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/marwan-elghitany-cntxt/rag-meter",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
