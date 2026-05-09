from setuptools import find_packages, setup


setup(
    name="distance-prediction-policy",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["setuptools"],
    zip_safe=True,
    description="Workspace-local policy using the vision distance prediction model",
)
