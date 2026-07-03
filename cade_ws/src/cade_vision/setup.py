from setuptools import setup, find_packages

setup(
    name='cade_vision',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        'cade_vision': [
            'kits/posture_gesture/models/lightgbm_pose_nan/model.txt',
            'kits/posture_gesture/models/lightgbm_pose_nan/feature_columns.json',
            'kits/posture_gesture/models/lightgbm_pose_nan_pose_only_with_p03_far_phone_lying/model.txt',
            'kits/posture_gesture/models/lightgbm_pose_nan_pose_only_with_p03_far_phone_lying/feature_columns.json',
            'kits/posture_gesture/models/lightgbm_gesture_static_yolo_pose/model.txt',
            'kits/posture_gesture/models/lightgbm_gesture_static_yolo_pose/feature_columns.json',
            'kits/posture_gesture/models/lightgbm_gesture_waving_motion_v2/model.txt',
            'kits/posture_gesture/models/lightgbm_gesture_waving_motion_v2/window_feature_columns.json',
            'kits/cloth/models/color_svm/color_model.xml',
            'kits/cloth/models/color_svm/color_model_norm.npz',
            'kits/cloth/models/color_svm/color_model_meta.json',
            'kits/cloth/models/color_svm/label_names.json',
            'models/yolo11n.pt',
            'models/yolo11s-fashionpedia-best.pt',
            'models/yolov8s-seg-fashionpedia-best.pt',
        ],
    },
    install_requires=[
        'rospy',
        'ultralytics',
        'opencv-python',
        'pyrealsense2',
        'numpy',
        'scipy',
        'lightgbm',
    ],
)
