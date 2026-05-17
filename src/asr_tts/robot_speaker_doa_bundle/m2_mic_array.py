import glob
import os
import subprocess
import sys
import time

from m2_doa_origin import estimate_doa_from_pcm


class M2MicArray:
    def __init__(self, serial_device="auto", baudrate=115200):
        self.serial_device = self.resolve_serial_device(serial_device)
        self.baudrate = int(baudrate)

    @staticmethod
    def resolve_serial_device(serial_device):
        if serial_device and serial_device != "auto":
            return serial_device

        preferred = ["/dev/wheeltec_mic"]
        candidates = preferred + sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
        for path in candidates:
            if os.path.exists(path):
                return path

        raise RuntimeError(
            "找不到麦克风阵列串口设备。请检查设备是否接入，或启动时显式传 --serial-device /dev/ttyACM0"
        )

    def _run_checked(self, cmd):
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip() or f"command failed with exit code {proc.returncode}"
            raise RuntimeError(detail)
        return proc

    def dump_origin_audio(self, out_dir, seconds=0.5):
        os.makedirs(out_dir, exist_ok=True)
        script = os.path.join(os.path.dirname(__file__), "m2_serial_cmd.py")
        py = sys.executable
        self._run_checked([py, script, "--device", self.serial_device, "--baud", str(self.baudrate), "--clean-pcm"])
        self._run_checked([py, script, "--device", self.serial_device, "--baud", str(self.baudrate), "--dump-start"])
        time.sleep(float(seconds))
        self._run_checked([py, script, "--device", self.serial_device, "--baud", str(self.baudrate), "--dump-stop"])
        self._run_checked([py, script, "--pull-audio", out_dir])
        return {
            "out_dir": os.path.abspath(out_dir),
            "origin_pcm": os.path.join(os.path.abspath(out_dir), "origin.pcm"),
            "iat_pcm": os.path.join(os.path.abspath(out_dir), "iat.pcm"),
        }

    def estimate_doa_from_origin(
        self,
        pcm_path,
        sample_rate=16000,
        total_channels=8,
        active_channels="0,1,2,3,4,5",
        geometry="circle6",
        num_mics=6,
        radius_m=0.035,
        spacing_m=0.028,
        angle_offset_deg=0.0,
        segment_seconds=0.5,
        angle_step_deg=2.0,
    ):
        return estimate_doa_from_pcm(
            pcm_path=pcm_path,
            sample_rate=sample_rate,
            total_channels=total_channels,
            active_channels=active_channels,
            geometry=geometry,
            num_mics=num_mics,
            radius_m=radius_m,
            spacing_m=spacing_m,
            angle_offset_deg=angle_offset_deg,
            segment_seconds=segment_seconds,
            angle_step_deg=angle_step_deg,
        )
