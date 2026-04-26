from setuptools import find_packages, setup


setup(
    name="data-gen-node",
    version="0.0.1",
    packages=find_packages(where="..", include=["data_gen_node", "data_gen_node.*"]),
    package_dir={"": ".."},
    install_requires=["setuptools"],
    zip_safe=True,
    description="Custom workspace-local policy package",
)
