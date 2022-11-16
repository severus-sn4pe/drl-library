from __future__ import annotations

from setuptools import find_packages
from setuptools import setup

# Read requirements.txt, ignore comments
try:
    REQUIRES = list()
    f = open("requirements.txt", "rb")
    for line in f.read().decode("utf-8").split("\n"):
        line = line.strip()
        if "#" in line:
            line = line[: line.find("#")].strip()
        if line:
            REQUIRES.append(line)
except FileNotFoundError:
    print("'requirements.txt' not found!")
    REQUIRES = list()

setup(
    name="drl",
    version="0.0.5",
    include_package_data=True,
    author="Renold Christian",
    author_email="christian.renold@hslu.ch",
    url="https://github.com/severus-sn4pe/drl-library",
    license="MIT",
    packages=find_packages(),
    install_requires=REQUIRES
    + ["pyfolio @ git+https://github.com/severus-sn4pe/pyfolio-reloaded.git@c797d6e7511a2139f2d5c4aff139db5285374c00"]
    + ["elegantrl @ git+https://github.com/AI4Finance-Foundation/ElegantRL.git#egg=elegantrl"],
    # install_requires=REQUIRES,
    description="Deep Reinforcement Learning based on FinRL",
    # long_description="A deep reinforcement Learning implementation based on FinRL",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Reinforcement Learning, Finance",
    platform=["any"],
    python_requires=">=3.7",
)