from setuptools import find_packages, setup

package_name = 'ais_stiffness_damping_calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='JungSeong',
    maintainer_email='jungseonglian@sju.ac.kr',
    description='Headless Cartesian stiffness/damping calibration sweep for AIC.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ais_stiffness_damping_calibration = ais_stiffness_damping_calibration.calibrate:main',
            'ais_stiffness_damping_visualize = ais_stiffness_damping_calibration.visualize:main',
            'ais_stiffness_damping_random_walk = ais_stiffness_damping_calibration.random_walk:main',
            'ais_stiffness_damping_hold_check = ais_stiffness_damping_calibration.hold_check:main',
        ],
    },
)
