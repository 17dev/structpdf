"""
structPDF Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="structpdf",
    version="0.1.0",
    author="Your Name",
    author_email="rimonhm@gmail.com",
    description="PDF extraction with",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/17dev/structpdf",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "dspy-ai>=2.5.0",
        "pymupdf>=1.23.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
        "openpyxl>=3.1.0",
        "pydantic>=2.0.0",
        "ipykernel>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "structpdf=structpdf.core:main",
        ],
    },
)
