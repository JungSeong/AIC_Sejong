from setuptools import find_packages, setup


setup(
    name="ais-reinforcement-learning",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["setuptools"],
    zip_safe=True,
    description="SFP-only reinforcement learning scaffolding for AIC insertion",
)
