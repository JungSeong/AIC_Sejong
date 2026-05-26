from setuptools import find_packages, setup


package_name = "ais_retry_classifier"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ],
    package_data={"": ["py.typed"]},
    install_requires=["setuptools", "numpy", "pyyaml"],
    zip_safe=True,
    maintainer="JungSeong",
    maintainer_email="jungseonglian@sju.ac.kr",
    description="SFP retry/success classifier dataset collection and baseline training tools.",
    license="TODO: License declaration",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "retry_capture_features = ais_retry_classifier.collection.feature_capture_node:main",
            "retry_make_scenarios = ais_retry_classifier.policies.scenario_plan:main",
            "retry_train_baseline = ais_retry_classifier.training.train_baseline:main",
        ],
    },
)
