from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ocrengine",  # Changed: removed hyphen for easier importing
    version="2.1.0",
    author="Miaomiao Yu",
    author_email="mmy@stanford.edu",
    description="OCR-Engine used as backend for external modules.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/miaomiaoyu",
    package_dir={"": "src"},  # Added: tell setuptools packages are in charnet/
    packages=find_packages(where="src"),  # Remove the include parameter
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,  # Shared dependencies
)
