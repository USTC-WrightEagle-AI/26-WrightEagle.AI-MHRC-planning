"""
姿态分类规则集合。

本模块只做几何规则判断，不直接调用 MediaPipe 或 RealSense：
- `landmarks` 来自 MediaPipe Pose，格式为 [(x, y, visibility), ...]。
  x/y 是 person crop 内的归一化 2D 坐标，y 轴向下。
- `keypoints_3d` 来自上游按 Pose 关键点从深度图反投影得到的 3D 坐标。
  当前规则沿用相机光学坐标约定：y 值越小，物理位置越高。

返回值统一为 `(posture, confidence)`：
- posture: "standing" / "sitting" / "lying" / "unknown"
- confidence: 规则置信度，不代表模型概率，只表示规则命中的强弱。
"""

import math

import numpy as np

# MediaPipe Pose 关键点索引常量。
# 这里只保留姿态分类需要的肩、髋、膝、踝，下游代码用这些索引访问
# `landmarks` 和 `keypoints_3d`，所以索引必须与 MediaPipe Pose 定义保持一致。
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


# ==================== 1. 2D 辅助与兜底模块 ====================


def _get_pixel_2d(landmarks, idx, img_width, img_height):
    """将归一化的比例坐标还原为真实的物理像素坐标"""
    return np.array([landmarks[idx][0] * img_width, landmarks[idx][1] * img_height])


def _is_visible_2d(landmarks, idx):
    """检查 2D Pose 点是否存在且可见性足够高。"""
    # 先检查长度，避免裁剪图中 Pose 输出异常或 landmark 数不足时越界。
    # 0.5 是本模块对“可用于姿态分类”的严格门槛；手势角度那边有更松的阈值。
    return len(landmarks) > idx and landmarks[idx][2] > 0.5


def _get_2d_vec(landmarks, idx):
    """把 MediaPipe 的归一化 x/y 坐标转成 numpy 向量，便于做角度和长度计算。"""
    return np.array([landmarks[idx][0], landmarks[idx][1]])


def _leg_angle_2d(landmarks, hip_idx, knee_idx, ankle_idx, img_width, img_height):
    """
    计算一条腿在 2D 投影里的两段夹角。

    注意：这里返回的是 knee->hip 与 knee->ankle 两个投影向量的夹角，
    不是严格的人体解剖学膝关节内角。后续阈值都是按这个定义调出来的，
    因此不要单独替换为另一种角度定义，否则 standing/sitting 的阈值也要重调。
    """
    if not all(_is_visible_2d(landmarks, i) for i in [hip_idx, knee_idx, ankle_idx]):
        return None

    hip = _get_pixel_2d(landmarks, hip_idx, img_width, img_height)
    knee = _get_pixel_2d(landmarks, knee_idx, img_width, img_height)
    ankle = _get_pixel_2d(landmarks, ankle_idx, img_width, img_height)

    # v1 表示大腿投影方向，v2 表示小腿投影方向。
    # 任一段长度接近 0 时，夹角没有稳定几何意义，直接返回 None。
    v1 = hip - knee
    v2 = ankle - knee
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None

    # clip 是为了抵消浮点误差，避免 acos 收到 1.00000001 这类非法值。
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))


def _compute_body_angles_3d(keypoints_3d: dict, v_spine_global) -> dict:
    """
    【数据提取与计算函数】
    计算 3D 空间下坐姿判定通道 A 所需的所有纯几何参数。
    - keypoints_3d 是从 mediapipe pose keypoints 后处理得到的关键点坐标
    - v_spine_global 为肩髋向量代表躯干向量
    """
    res = {
        "lean_angle": None,
        "left_joint_angle": None,
        "left_thigh_ratio": None,
        "right_joint_angle": None,
        "right_thigh_ratio": None,
    }

    # 1. 计算脊柱偏离竖直轴的绝对倾斜角
    vsg_norm = np.linalg.norm(v_spine_global)
    cos_lean = abs(v_spine_global[1]) / vsg_norm
    res["lean_angle"] = math.degrees(math.acos(np.clip(cos_lean, 0.0, 1.0)))

    # 2. 左右侧单侧自适应空间向量解算
    left_valid = all(keypoints_3d.get(i) is not None for i in (11, 23, 25))
    right_valid = all(keypoints_3d.get(i) is not None for i in (12, 24, 26))

    # 左侧解算
    if left_valid:
        # 统一以 髋关节(23) 为圆心发射
        v_torso = np.array(keypoints_3d[11]) - np.array(keypoints_3d[23])  # 髋 -> 肩
        v_thigh = np.array(keypoints_3d[25]) - np.array(keypoints_3d[23])  # 髋 -> 膝
        vt_n, vth_n = np.linalg.norm(v_torso), np.linalg.norm(v_thigh)

        if vt_n > 1e-6 and vth_n > 1e-6:
            cos_joint = np.dot(v_torso, v_thigh) / (vt_n * vth_n)
            res["left_joint_angle"] = math.degrees(
                math.acos(np.clip(cos_joint, -1.0, 1.0))
            )
            res["left_thigh_ratio"] = abs(v_thigh[1]) / vth_n

    # 右侧解算
    if right_valid:
        # 统一以 髋关节(24) 为圆心发射
        v_torso = np.array(keypoints_3d[12]) - np.array(keypoints_3d[24])  # 跨 -> 肩
        v_thigh = np.array(keypoints_3d[26]) - np.array(keypoints_3d[24])  # 跨 -> 膝
        vt_n, vth_n = np.linalg.norm(v_torso), np.linalg.norm(v_thigh)

        if vt_n > 1e-6 and vth_n > 1e-6:
            cos_joint = np.dot(v_torso, v_thigh) / (vt_n * vth_n)
            res["right_joint_angle"] = math.degrees(
                math.acos(np.clip(cos_joint, -1.0, 1.0))
            )
            res["right_thigh_ratio"] = abs(v_thigh[1]) / vth_n

    return res


def _compute_sitting_features_b(
    landmarks, keypoints_3d: dict, img_width, img_height
) -> dict:
    """
    【数据解算器 - 通道 B】
    统一抽取 3D 垂直落差特征与 2D 物理像素下的透视缩水特征。
    """
    features = {
        "thigh_drop_3d": None,  # 髋膝高度差
        "calf_drop_3d": None,  # 膝踝落差（小腿下垂度）
        "thigh_torso_ratio_2d": None,
    }

    # 1. 3D 物理高度提取（Y 轴，单位：米）
    def get_h3d(idx1, idx2):
        p = keypoints_3d.get(idx1) or keypoints_3d.get(idx2)
        return p[1] if p is not None else None

    hi_y = get_h3d(23, 24)  # 髋关节
    kn_y = get_h3d(25, 26)  # 膝盖
    an_y = get_h3d(27, 28)  # 脚踝
    sh_y = get_h3d(11, 12)  # 肩膀

    if all(y is not None for y in (hi_y, kn_y, an_y, sh_y)):
        features["thigh_drop_3d"] = abs(kn_y - hi_y)  # 髋膝高度差
        features["calf_drop_3d"] = an_y - kn_y  # 膝踝落差（小腿下垂度）

    # 2. 2D 像素平面投影缩水比值计算（使用已脱敏的 _get_pixel_2d）
    left_2d = all(len(landmarks) > i and landmarks[i][2] > 0.5 for i in (11, 23, 25))
    right_2d = all(len(landmarks) > i and landmarks[i][2] > 0.5 for i in (12, 24, 26))

    if left_2d or right_2d:
        h_idx, k_idx, s_idx = (11, 23, 25) if left_2d else (12, 24, 26)

        # 还原到各项同性的物理像素空间计算欧氏距离
        torso_len_2d = np.linalg.norm(
            _get_pixel_2d(landmarks, s_idx, img_width, img_height)
            - _get_pixel_2d(landmarks, h_idx, img_width, img_height)
        )
        thigh_len_2d = np.linalg.norm(
            _get_pixel_2d(landmarks, k_idx, img_width, img_height)
            - _get_pixel_2d(landmarks, h_idx, img_width, img_height)
        )

        if torso_len_2d > 1e-6:
            features["thigh_torso_ratio_2d"] = thigh_len_2d / torso_len_2d

    return features


# ==================== 2. 3D 核心几何子规则隔离（易改区） ====================


def is_torso_vertical_3d(keypoints_3d: dict):
    """
    【基础前置规则】躯干竖直判定。

    返回:
        (torso_ok, v_spine_global)
    - torso_ok 代表躯干是否竖直
    - v_spine_global 为肩髋向量代表躯干向量
    判定思路:
    1. 如果双肩 + 双髋都存在，用肩线和脊柱方向构造躯干平面；
       平面法向量与重力方向接近垂直时，说明躯干面大致竖直。
    2. 如果只拿到单侧肩髋，则退化成“肩到髋向量是否接近竖直”的判断。

    `v_spine_global` 会继续给 sitting 规则复用，避免重复计算。
    """
    # s 代表肩膀，h 代表髋
    ls, rs, lh, rh = (
        keypoints_3d.get(11),
        keypoints_3d.get(12),
        keypoints_3d.get(23),
        keypoints_3d.get(24),
    )
    v_spine_global = None

    if all(v is not None for v in (ls, rs, lh, rh)):
        # 双侧关键点齐全时，使用肩中点到髋中点作为近似脊柱方向；
        # 再用双肩连线和脊柱方向叉乘，得到躯干平面的法向量。
        shoulder_mid = (np.array(ls) + np.array(rs)) / 2
        hip_mid = (np.array(lh) + np.array(rh)) / 2
        v1 = np.array(rs) - np.array(ls)
        v2 = shoulder_mid - hip_mid
        v_spine_global = v2

        n = np.cross(v1, v2)
        gravity = np.array([0.0, -1.0, 0.0])
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-6:
            # 使用 abs 是为了不关心法向量朝向，只关心躯干平面相对重力的夹角。
            # 65~115 度是一个较宽的“近似竖直平面”窗口，给深度噪声留余量。
            cos_theta = abs(np.dot(n, gravity)) / n_norm
            theta = math.degrees(math.acos(np.clip(cos_theta, 0.0, 1.0)))
            return (65 <= theta <= 115), v_spine_global

    # 双侧点不完整时退化到单侧肩髋向量。这个分支精度低一些，
    # 但能覆盖侧身、遮挡或部分关键点深度失效的情况。
    if ls is not None and lh is not None:
        v_spine_global = np.array(ls) - np.array(lh)
    elif rs is not None and rh is not None:
        v_spine_global = np.array(rs) - np.array(rh)

    if v_spine_global is not None:
        vs_norm = np.linalg.norm(v_spine_global)
        if vs_norm > 1e-6:
            # spine_angle 是脊柱向量偏离竖直方向的角度。
            # 小于 25 度时认为躯干足够直立，可作为 standing 的前置条件。
            cos_spine = abs(v_spine_global[1]) / vs_norm
            spine_angle = math.degrees(math.acos(np.clip(cos_spine, 0.0, 1.0)))
            return (spine_angle < 25), v_spine_global

    return False, None


def check_sitting_3d(
    right_leg_angle, left_leg_angle, body_angles_3d: dict, features_b: dict
) -> bool:
    """
    【完全体 - 坐姿规则判定模块】
    摒弃所有底层复杂的坐标和距离运算，只专注于三大规则分支的互斥与联动。
    """
    # 🌟 规则零：免死金牌一票否决
    # 如果 2D 像素角度算出来你双腿笔直，无论 3D 深度怎么晃动，一律放行站立，绝不触发坐姿拦截。
    if any(a is not None and a >= 150 for a in (right_leg_angle, left_leg_angle)):
        return False

    # ==========================================
    # 🚀 通道 A：斜向大角度后靠/前倾规则过滤器
    # ==========================================
    lean_angle = body_angles_3d.get("lean_angle")
    if lean_angle is not None and lean_angle < 50:
        # 左侧校验
        if body_angles_3d["left_joint_angle"] is not None:
            if 60 <= body_angles_3d["left_joint_angle"] <= 125:
                if body_angles_3d["left_thigh_ratio"] < 0.642:
                    return "channel A sitting"
        # 右侧校验
        if body_angles_3d["right_joint_angle"] is not None:
            if 60 <= body_angles_3d["right_joint_angle"] <= 125:
                if body_angles_3d["right_thigh_ratio"] < 0.642:
                    return "channel A sitting"

    # ==========================================
    # 🚀 通道 B：物理高度差与 2D 投影缩水规则过滤器
    # ==========================================
    thigh_drop_3d = features_b.get("thigh_drop_3d")
    calf_drop_3d = features_b.get("calf_drop_3d")
    thigh_torso_ratio_2d = features_b.get("thigh_torso_ratio_2d")

    if thigh_drop_3d is not None and calf_drop_3d is not None:
        # 特征1：髋关节与膝盖高度极其平齐（垂直落差小于 10cm）
        # 特征2：膝盖到脚踝有明显的垂直落差（小腿垂直自然落地，高差大于 15cm）
        if thigh_drop_3d < 0.10 and calf_drop_3d > 0.15:
            # 特征3：大腿在 2D 像素投影上相对躯干极度变短（正面透视缩水比例 < 0.42）
            if thigh_torso_ratio_2d is not None and thigh_torso_ratio_2d < 0.42:
                return "channel B sitting"

    return False


def check_standing_3d(
    keypoints_3d: dict, torso_ok: bool, left_angle, right_angle
) -> bool:
    """
    【站立姿态模块】结合躯干形态、运动学膝关节角度和严格的 3D 物理高差链。

    站立判断故意做得偏严格：
    1. 躯干必须通过 `is_torso_vertical_3d()`。
    2. 至少一条腿的 2D 投影角度满足“接近站立腿形”的经验阈值。
    3. 3D 高度链必须满足肩高于髋、膝低于髋、踝低于膝。

    这样可以减少坐姿、半蹲、弯腰时被误判为 standing。
    """
    # 1. 2D 膝关节运动学前置校验
    valid_angles = [a for a in (left_angle, right_angle) if a is not None]
    angles_str = "\n".join([str(a) for a in valid_angles])
    knee_ok = any(a >= 145 for a in valid_angles) if valid_angles else False

    if not torso_ok:
        return "torso_false"
    if not knee_ok:
        if valid_angles is not None:
            return f"{angles_str}"
        else:
            return "valid_angles is none"

    # if not (torso_ok and knee_ok):
    #     return False

    # 2. 严格 3D 物理高度链校验
    def get_h3d(idx1, idx2):
        # 站立高度链只需要左右同类关键点中至少一侧可用。
        # 这能减轻人体侧身、衣物遮挡、深度空洞对判断的影响。
        p = keypoints_3d.get(idx1) or keypoints_3d.get(idx2)
        return p[1] if p is not None else None

    sh_y = get_h3d(11, 12)
    hi_y = get_h3d(23, 24)
    kn_y = get_h3d(25, 26)
    an_y = get_h3d(27, 28)

    if all(y is not None for y in (sh_y, hi_y, kn_y, an_y)):
        thigh_drop = kn_y - hi_y
        # 站立约束：
        # - sh_y < hi_y - 0.05: 肩点比髋点高至少 5cm；
        # - thigh_drop > 0.18: 髋到膝有明显垂直落差，排除坐姿髋膝齐平；
        # - kn_y < an_y - 0.05: 膝点比踝点高至少 5cm，形成完整下肢高度链。
        if (sh_y < hi_y - 0.05) and (thigh_drop > 0.18) and (kn_y < an_y - 0.05):
            # return True
            return "standing\n" + f"{angles_str}"

    # return False
    return "thigh_drop 硬编码太严格"


def check_lying_3d(keypoints_3d: dict) -> bool:
    """
    【平躺姿态模块】独立隔离。未来若要加入 2D 横向脊柱规则，在此直接扩展即可。

    当前 3D 规则只看高度压缩：肩、髋、膝在竖直轴上的 spread 很小，
    说明这些身体点处在近似同一高度平面，更接近平躺/倒地状态。
    """

    def get_h3d(idx1, idx2):
        # 左右同类关键点取任一可用值，保持对局部遮挡的容错。
        p = keypoints_3d.get(idx1) or keypoints_3d.get(idx2)
        return p[1] if p is not None else None

    sh_y = get_h3d(11, 12)
    hi_y = get_h3d(23, 24)
    kn_y = get_h3d(25, 26)

    # 核心几何特征：人在躺着时，肩、髋、膝在相机 Y 轴（高度轴）上极度收缩紧凑
    if all(y is not None for y in [sh_y, hi_y, kn_y]):
        spread = max(sh_y, hi_y, kn_y) - min(sh_y, hi_y, kn_y)
        # 25cm 是比较宽松的跌倒/平躺高度压缩阈值。它不要求人体完全水平，
        # 目的是覆盖倒地、斜躺、局部关键点噪声等真实场景。
        if spread < 0.25:  # 垂直高差高度压缩在 25cm 以内
            return True

    return False


# ==================== 3. 主入口决策树（平行排他分支） ====================


def classify_posture_3d(landmarks, img_width, img_height, keypoints_3d: dict = None):
    """
    解耦后的高清晰度主函数入口，各姿态逻辑高度自平衡、互不干扰。

    Args:
        landmarks: MediaPipe Pose 2D landmarks。
        img_height: 兼容旧接口保留，目前本规则模块不直接使用。
        keypoints_3d: 由上游深度图采样得到的 3D 关键点字典。

    决策顺序说明:
    - 先判 sitting，因为坐姿特征相对明确且容易被站立规则吞掉。
    - 再判 standing，并在 standing 命中后用 lying 做一次高度压缩反查。
    - 最后用 lying 兜底，捕获未满足坐/站但高度明显压缩的倒地状态。
    """
    if not keypoints_3d or not any(p is not None for p in keypoints_3d.values()):
        return "unknown", 0.0

    # 预计算跨模块复用的几何基础特征
    # torso_ok 用作 standing 的前置条件；v_spine_global 供 sitting 通道 A 复用。
    # 左右腿 2D 角度既用于 2D fallback，也作为 3D standing 的轻量运动学约束。
    torso_ok, v_spine_global = is_torso_vertical_3d(keypoints_3d)
    left_leg_angle = _leg_angle_2d(
        landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, img_width, img_height
    )
    right_leg_angle = _leg_angle_2d(
        landmarks,
        RIGHT_HIP,
        RIGHT_KNEE,
        RIGHT_ANKLE,
        img_width,
        img_height,
    )
    body_angles_3d = _compute_body_angles_3d(keypoints_3d, v_spine_global)
    features_b = _compute_sitting_features_b(
        landmarks, keypoints_3d, img_width, img_height
    )

    # 3. 三大姿态独立规则处理器并行诊断（按优先级或排他逻辑拦截）

    # 优先级 A：坐姿拦截（双通道几何条件非常明确，不易误触）
    sitting_result = check_sitting_3d(
        right_leg_angle, left_leg_angle, body_angles_3d, features_b
    )
    if sitting_result:
        return sitting_result, 0.85

    # # 优先级 B：站立拦截（依赖高度链与直立刚性约束）
    # if check_standing_3d(keypoints_3d, torso_ok, left_angle, right_angle):
    #     # 在原逻辑中，满足了站立条件但高度极度压缩时，实际上是躺倒状态
    #     if check_lying_3d(keypoints_3d):
    #         return "lying", 0.85
    #     return "standing", 0.8
    standing_result = check_standing_3d(
        keypoints_3d, torso_ok, left_leg_angle, right_leg_angle
    )
    if standing_result:
        return standing_result, 0.8

    # 优先级 C：防漏跌倒/平躺拦截（未能通过严格站立和坐姿、但高度极度压缩的特殊状态兜底）
    if check_lying_3d(keypoints_3d):
        return "lying", 0.75  # 纯兜底触发，置信度略调低

    # 没有任何高置信规则命中时不强行猜测，避免把遮挡/深度缺失场景误报为确定姿态。
    return "unknown", 0.3
