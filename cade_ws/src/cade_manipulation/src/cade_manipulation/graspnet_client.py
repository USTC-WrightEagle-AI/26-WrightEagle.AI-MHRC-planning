"""
GraspNet TCP Client — communicates with the 5090 GraspNet server.

Wire protocol (length-prefixed binary over TCP port 9090):

  Robot → 5090:
    1. JSON bytes:  {"text_prompt": "cup"}
    2. RGB image:   PNG-encoded 8-bit color frame
    3. Depth image: PNG-encoded 16-bit depth frame (CRITICAL: must be 16-bit)

  5090 → Robot:
    {
      "success": true,
      "error_code": "OK",
      "message": "...",
      "grasp_poses": {
        "position":    [x, y, z],
        "orientation": [x, y, z, w]
      }
    }

Each message is prefixed with an 8-byte big-endian length header.
"""

import json
import socket
import struct
from typing import Optional

import numpy as np

# cv2 is imported lazily — only the PNG encode/decode paths need it.
try:
    import cv2
except ImportError:
    cv2 = None


class GraspNetClient:
    """Thin client for the 5090 GraspNet TCP socket server."""

    def __init__(
        self,
        host: str = "192.168.1.100",
        port: int = 9090,
        timeout: float = 10.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query_grasp(
        self,
        text_prompt: str,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
    ) -> dict:
        """
        Send a grasp request to the 5090 server.

        Args:
            text_prompt:  Object description, e.g. "cup", "bottle".
            rgb_image:    BGR image (H, W, 3) as uint8 numpy array.
            depth_image:  16-bit depth image (H, W) as uint16 numpy array.
                          Each unit = 1 mm (RealSense default).

        Returns:
            dict with keys:
              success:     bool
              error_code:  str (e.g. "OK", "NO_GRASP", "TIMEOUT")
              message:     human-readable status
              grasp_poses: {"position": [x,y,z], "orientation": [x,y,z,w]} | None
        """
        if cv2 is None:
            return self._error("CV2_NOT_AVAILABLE", "opencv-python is not installed")

        self._validate_images(rgb_image, depth_image)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))

            # 1. JSON prompt
            json_payload = json.dumps({"text_prompt": text_prompt}).encode("utf-8")
            self._send_msg(sock, json_payload)

            # 2. RGB PNG
            _, rgb_png = cv2.imencode(".png", rgb_image)
            self._send_msg(sock, rgb_png.tobytes())

            # 3. 16-bit depth PNG
            _, depth_png = cv2.imencode(".png", depth_image)
            self._send_msg(sock, depth_png.tobytes())

            # 4. Receive result
            result_bytes = self._recv_msg(sock)
            sock.close()

            return json.loads(result_bytes.decode("utf-8"))

        except socket.timeout:
            return self._error("TIMEOUT", "GraspNet server did not respond in time")
        except ConnectionRefusedError:
            return self._error(
                "CONNECTION_REFUSED",
                f"GraspNet server not reachable at {self.host}:{self.port}",
            )
        except OSError as exc:
            return self._error("SOCKET_ERROR", str(exc))
        except json.JSONDecodeError as exc:
            return self._error("BAD_RESPONSE", f"Invalid JSON from server: {exc}")

    def query_grasp_from_files(
        self,
        text_prompt: str,
        rgb_path: str,
        depth_path: str,
    ) -> dict:
        """
        Convenience wrapper: read RGB + depth from disk, then call query_grasp.

        Depth image MUST be read with cv2.IMREAD_UNCHANGED to preserve 16-bit.
        """
        if cv2 is None:
            return self._error("CV2_NOT_AVAILABLE", "opencv-python is not installed")

        rgb = cv2.imread(rgb_path)
        if rgb is None:
            return self._error("BAD_IMAGE", f"Cannot read RGB image: {rgb_path}")

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            return self._error("BAD_IMAGE", f"Cannot read depth image: {depth_path}")

        if depth.dtype != np.uint16:
            return self._error(
                "BAD_DEPTH",
                f"Depth must be 16-bit, got {depth.dtype}. "
                "Use cv2.IMREAD_UNCHANGED when loading.",
            )

        return self.query_grasp(text_prompt, rgb, depth)

    # ------------------------------------------------------------------
    # Wire protocol helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _send_msg(sock: socket.socket, data: bytes) -> None:
        """Send length-prefixed binary message."""
        sock.sendall(struct.pack(">Q", len(data)) + data)

    @staticmethod
    def _recv_msg(sock: socket.socket) -> bytes:
        """Receive length-prefixed binary message."""
        header = sock.recv(8)
        if len(header) < 8:
            raise OSError("Server closed connection before sending response header")
        msg_len = struct.unpack(">Q", header)[0]
        chunks = []
        remaining = msg_len
        while remaining > 0:
            chunk = sock.recv(min(remaining, 4096))
            if not chunk:
                raise OSError("Server closed connection mid-response")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_images(rgb: np.ndarray, depth: np.ndarray) -> None:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"RGB must be HxWx3, got shape {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError(f"RGB must be uint8, got {rgb.dtype}")
        if depth.ndim != 2:
            raise ValueError(f"Depth must be HxW, got shape {depth.shape}")
        if depth.dtype != np.uint16:
            raise ValueError(
                f"Depth must be uint16 (millimetre units), got {depth.dtype}. "
                "8-bit depth images will silently break GraspNet results."
            )

    @staticmethod
    def _error(code: str, message: str) -> dict:
        return {
            "success": False,
            "error_code": code,
            "message": message,
            "grasp_poses": None,
        }
