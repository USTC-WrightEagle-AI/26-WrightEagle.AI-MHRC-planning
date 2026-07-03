from setuptools import find_packages, setup

setup(
    name="cade_manipulation",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "rospy",
        "numpy",
        "opencv-python",
        "pyrealsense2",
    ],
)
