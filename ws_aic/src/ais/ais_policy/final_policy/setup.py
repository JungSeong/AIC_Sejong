from setuptools import find_packages, setup


package_name = "final_policy"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "torch", "huggingface-hub"],
    zip_safe=True,
    maintainer="JungSeong",
    maintainer_email="jungseonglian@sju.ac.kr",
    description="Final AIS policy using unified pose prediction.",
    license="TODO: License declaration",
)
