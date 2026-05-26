from setuptools import find_packages, setup


package_name = "ais_ground_truth_guided_residual_visual_servoing"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ],
    package_data={"": ["py.typed"]},
    install_requires=["setuptools", "numpy", "pyyaml", "ais-transform"],
    zip_safe=True,
    maintainer="JungSeong",
    maintainer_email="jungseonglian@sju.ac.kr",
    description="SFP ground-truth-guided residual visual servoing training package.",
    license="TODO: License declaration",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "grvs_make_sfp_batch_config = ais_ground_truth_guided_residual_visual_servoing.batch.engine_config:main",
            "grvs_collect_batch = ais_ground_truth_guided_residual_visual_servoing.batch.collect_batch:main",
            "grvs_test_batch = ais_ground_truth_guided_residual_visual_servoing.batch.test_batch:main",
            "grvs_batch_round = ais_ground_truth_guided_residual_visual_servoing.batch.batch_round:main",
            "grvs_capture_node = ais_ground_truth_guided_residual_visual_servoing.capture_node:main",
            "grvs_train_distance = ais_ground_truth_guided_residual_visual_servoing.training.train_distance:main",
            "grvs_train_rotation = ais_ground_truth_guided_residual_visual_servoing.training.train_rotation:main",
            "grvs_train_joint = ais_ground_truth_guided_residual_visual_servoing.training.train_joint:main",
        ],
    },
)
