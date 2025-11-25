"""
Setup script for SafetyKnob package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="safetyknob",
    version="2.0.0",
    author="SafetyKnob Team",
    author_email="team@safetyknob.ai",
    description="Multi-model safety classification system for images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/safetyknob",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.9b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "api": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "python-multipart>=0.0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "safetyknob-analyze=scripts.run_analysis:main",
            "safetyknob-predict=scripts.predict:main",
            "safetyknob-monitor=scripts.monitor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "safetyknob": ["config/*.json", "data/*.txt"],
    },
)