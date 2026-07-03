"""Pose parsing, angle conversion, and TF helpers for cade_navigation."""

import json
import math
import re
from typing import Any, Iterable, Tuple


PositionTuple = Tuple[float, float, float]
Matrix4 = Tuple[Tuple[float, float, float, float], ...]


def normalize_angle_deg(angle: float) -> float:
    """Normalize an angle in degrees to [-180, 180)."""
    while angle >= 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def yaw_deg_to_quaternion(yaw_deg: float):
    """Convert yaw in degrees to a geometry-compatible quaternion tuple."""
    from tf.transformations import quaternion_from_euler

    yaw_rad = math.radians(float(yaw_deg))
    return quaternion_from_euler(0.0, 0.0, yaw_rad)


def quaternion_to_yaw_deg(quaternion: Iterable[float]) -> float:
    """Convert a quaternion iterable into normalized yaw degrees."""
    from tf.transformations import euler_from_quaternion

    _, _, yaw = euler_from_quaternion(tuple(quaternion))
    return normalize_angle_deg(math.degrees(yaw))


def parse_position(value: Any, field_name: str = "position") -> PositionTuple:
    """
    Parse a position from list/tuple, dict, Pydantic-like object, or string.

    Accepted string examples:
      - "1.0, 2.0, 0.0"
      - "1.0 2.0 0.0"
      - "[1.0, 2.0, 0.0]"
      - "{\"x\": 1.0, \"y\": 2.0, \"z\": 0.0}"
    """
    if value is None:
        raise ValueError("Missing required parameter: %s" % field_name)

    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif all(hasattr(value, attr) for attr in ("x", "y")):
        value = {
            "x": getattr(value, "x"),
            "y": getattr(value, "y"),
            "z": getattr(value, "z", 0.0),
        }

    if isinstance(value, dict):
        try:
            return (
                float(value["x"]),
                float(value["y"]),
                float(value.get("z", 0.0)),
            )
        except KeyError as exc:
            raise ValueError("%s dict must contain x and y" % field_name) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("%s dict values must be numeric" % field_name) from exc

    if isinstance(value, (list, tuple)):
        return _parse_numeric_sequence(value, field_name)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("%s cannot be empty" % field_name)

        if text[0] in "[{":
            try:
                return parse_position(json.loads(text), field_name=field_name)
            except json.JSONDecodeError:
                pass

        parts = [item for item in re.split(r"[\s,;]+", text) if item]
        return _parse_numeric_sequence(parts, field_name)

    raise ValueError(
        "%s must be [x, y, z], {'x': x, 'y': y, 'z': z}, or a coordinate string"
        % field_name
    )


def _parse_numeric_sequence(values: Iterable[Any], field_name: str) -> PositionTuple:
    items = list(values)
    if len(items) not in (2, 3):
        raise ValueError("%s must contain 2 or 3 numeric values" % field_name)

    try:
        x = float(items[0])
        y = float(items[1])
        z = float(items[2]) if len(items) == 3 else 0.0
    except (TypeError, ValueError) as exc:
        raise ValueError("%s values must be numeric" % field_name) from exc
    return x, y, z


def parse_float(value: Any, default: float, field_name: str) -> float:
    """Parse an optional numeric field with a clear error."""
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % field_name) from exc


def parse_matrix4(value: Any, field_name: str = "matrix") -> Matrix4:
    """Parse a 4x4 transform matrix from a nested list, flat list, or string."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("%s cannot be empty" % field_name)
        if text[0] in "[{":
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("%s must be valid JSON" % field_name) from exc
        else:
            value = [item for item in re.split(r"[\s,;]+", text) if item]

    if not isinstance(value, (list, tuple)):
        raise ValueError("%s must be a 4x4 list or a flat 16-value list" % field_name)

    rows = list(value)
    if len(rows) == 4 and all(isinstance(row, (list, tuple)) for row in rows):
        matrix_rows = [list(row) for row in rows]
        if any(len(row) != 4 for row in matrix_rows):
            raise ValueError("%s must contain four rows of four values" % field_name)
    elif len(rows) == 16:
        matrix_rows = [rows[index:index + 4] for index in range(0, 16, 4)]
    else:
        raise ValueError("%s must be 4x4 or contain 16 values" % field_name)

    try:
        return tuple(
            tuple(float(value) for value in row)
            for row in matrix_rows
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("%s values must be numeric" % field_name) from exc


def apply_matrix4(matrix: Matrix4, point_xyz: PositionTuple) -> PositionTuple:
    """Apply a homogeneous 4x4 transform matrix to a 3D point."""
    x, y, z = [float(value) for value in point_xyz]
    vector = (x, y, z, 1.0)
    result = [
        sum(float(matrix[row][col]) * vector[col] for col in range(4))
        for row in range(4)
    ]
    w = result[3]
    if abs(w) > 1e-9 and abs(w - 1.0) > 1e-9:
        return result[0] / w, result[1] / w, result[2] / w
    return result[0], result[1], result[2]


def transform_point(tf_buffer, point_xyz: PositionTuple, source_frame: str,
                    target_frame: str, timeout: float = 1.0) -> PositionTuple:
    """Transform a point between TF frames."""
    if source_frame == target_frame:
        return float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])

    import rospy

    transform = tf_buffer.lookup_transform(
        target_frame,
        source_frame,
        rospy.Time(0),
        rospy.Duration(float(timeout)),
    )
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    rotated_x, rotated_y, rotated_z = rotate_vector_by_quaternion(
        (
            float(point_xyz[0]),
            float(point_xyz[1]),
            float(point_xyz[2]),
        ),
        (rotation.x, rotation.y, rotation.z, rotation.w),
    )
    return (
        float(translation.x) + rotated_x,
        float(translation.y) + rotated_y,
        float(translation.z) + rotated_z,
    )


def rotate_vector_by_quaternion(vector: PositionTuple,
                                quaternion: Iterable[float]) -> PositionTuple:
    """Rotate a 3D vector by quaternion without PyKDL/tf2_geometry_msgs."""
    x, y, z = vector
    qx, qy, qz, qw = [float(value) for value in quaternion]

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return float(x), float(y), float(z)

    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    # Rodrigues-equivalent quaternion vector rotation:
    # v' = v + 2*w*(q_vec x v) + 2*(q_vec x (q_vec x v)).
    uv_x = qy * z - qz * y
    uv_y = qz * x - qx * z
    uv_z = qx * y - qy * x

    uuv_x = qy * uv_z - qz * uv_y
    uuv_y = qz * uv_x - qx * uv_z
    uuv_z = qx * uv_y - qy * uv_x

    return (
        float(x + 2.0 * (qw * uv_x + uuv_x)),
        float(y + 2.0 * (qw * uv_y + uuv_y)),
        float(z + 2.0 * (qw * uv_z + uuv_z)),
    )


def lookup_pose(tf_buffer, target_frame: str, source_frame: str,
                timeout: float = 1.0) -> Tuple[float, float, float]:
    """Return source frame pose as x, y, yaw_deg in target frame."""
    import rospy

    transform = tf_buffer.lookup_transform(
        target_frame,
        source_frame,
        rospy.Time(0),
        rospy.Duration(float(timeout)),
    )
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    yaw_deg = quaternion_to_yaw_deg(
        (rotation.x, rotation.y, rotation.z, rotation.w)
    )
    return float(translation.x), float(translation.y), yaw_deg


def read_current_map_pose(map_frame: str = "map", base_frame: str = "base_link_fusion",
                          timeout: float = 5.0) -> Tuple[float, float, float]:
    """Read the current robot pose from the classic tf listener."""
    import rospy
    import tf

    listener = tf.TransformListener()
    listener.waitForTransform(
        map_frame,
        base_frame,
        rospy.Time(0),
        rospy.Duration(float(timeout)),
    )
    trans, rot = listener.lookupTransform(map_frame, base_frame, rospy.Time(0))
    return float(trans[0]), float(trans[1]), quaternion_to_yaw_deg(rot)
