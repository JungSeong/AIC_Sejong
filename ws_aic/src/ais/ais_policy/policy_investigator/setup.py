from setuptools import find_packages, setup

package_name = "policy_investigator"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="swlinux",
    maintainer_email="swlinux@todo.todo",
    description="Runtime visual investigator for AIC policies.",
    license="TODO: License declaration",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "policy_investigator_gui = policy_investigator.gui:main",
        ],
    },
)
