from setuptools import find_packages, setup


package_name = "ais_pose_prediction"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    package_data={"": ["py.typed"]},
    install_requires=["setuptools", "numpy", "torch", "torchvision"],
    zip_safe=True,
    maintainer="JungSeong",
    maintainer_email="jungseonglian@sju.ac.kr",
    description="Unified pose prediction policy for AIS port alignment.",
    license="TODO: License declaration",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "ais_pose_prediction_train = ais_pose_prediction.train:main",
        ],
    },
)
