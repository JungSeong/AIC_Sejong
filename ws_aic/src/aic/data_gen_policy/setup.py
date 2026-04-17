from setuptools import find_packages, setup


setup(
    name="data-gen-policy",
    version="0.0.1",
    packages=find_packages(where="..", include=["data_gen_policy", "data_gen_policy.*"]),
    package_dir={"": ".."},
    install_requires=["setuptools"],
    zip_safe=True,
    description="Custom workspace-local policy package",
)
