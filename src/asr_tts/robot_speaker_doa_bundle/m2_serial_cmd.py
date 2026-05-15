import argparse
import json
import os
import shutil
import subprocess
import time


def packetize_json(payload):
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    sync_head = 0xA5
    user_id = 0x01
    msg_type = 0x05
    msg_id_l = 0x01
    msg_id_h = 0x00
    msg_len = len(content)
    msg_len_h = (msg_len >> 8) & 0xFF
    msg_len_l = msg_len & 0xFF
    checksum = (
        (~sum([sync_head, user_id, msg_type, msg_len_l, msg_len_h, msg_id_l, msg_id_h] + list(content))) & 0xFF
    ) + 1
    return bytes([sync_head, user_id, msg_type, msg_len_l, msg_len_h, msg_id_l, msg_id_h]) + content + bytes(
        [checksum & 0xFF]
    )


def setup_tty(fd, baudrate):
    import termios

    attrs = termios.tcgetattr(fd)
    attrs[2] = attrs[2] | termios.CREAD | termios.CLOCAL
    attrs[2] = attrs[2] & ~termios.PARENB
    attrs[2] = attrs[2] & ~termios.CSTOPB
    attrs[2] = attrs[2] & ~termios.CSIZE
    attrs[2] = attrs[2] | termios.CS8
    attrs[3] = 0
    attrs[1] = 0
    attrs[0] = 0
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    speed = getattr(termios, f"B{int(baudrate)}", None)
    if speed is not None:
        attrs[4] = speed
        attrs[5] = speed
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def resolve_adb():
    adb = os.environ.get("ADB") or shutil.which("adb")
    if adb:
        return adb
    raise SystemExit("adb not found. 请先安装 adb，或设置环境变量 ADB=/path/to/adb")


def send_payload(device, baudrate, payload, timeout=2.0):
    fd = os.open(device, os.O_RDWR | os.O_NOCTTY)
    try:
        setup_tty(fd, baudrate)
        os.write(fd, packetize_json(payload))
        end = time.time() + float(timeout)
        raw = bytearray()
        while time.time() < end:
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                time.sleep(0.01)
                continue
            if chunk:
                raw += chunk
            else:
                time.sleep(0.01)
        return raw
    finally:
        os.close(fd)


def adb_pull_audio(out_dir):
    adb = resolve_adb()
    os.makedirs(out_dir, exist_ok=True)
    pulled = []
    for remote_name in ("/data/build/origin.pcm", "/data/build/iat.pcm"):
        local_name = os.path.join(out_dir, os.path.basename(remote_name))
        subprocess.run([adb, "pull", remote_name, local_name], check=True, text=True)
        pulled.append(local_name)
    return pulled


def main():
    parser = argparse.ArgumentParser(description="M2 serial command helper")
    parser.add_argument("--device", default="/dev/wheeltec_mic")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--clean-pcm", action="store_true")
    parser.add_argument("--dump-start", action="store_true")
    parser.add_argument("--dump-stop", action="store_true")
    parser.add_argument("--pull-audio", default=None)
    args = parser.parse_args()

    payloads = []
    if args.version:
        payloads.append({"type": "version"})
    if args.clean_pcm:
        payloads.append({"type": "clean_pcm"})
    if args.dump_start:
        payloads.append({"type": "dump_audio", "content": {"debug": 3}})
    if args.dump_stop:
        payloads.append({"type": "dump_audio", "content": {"debug": 0}})

    if payloads and not os.path.exists(args.device):
        raise SystemExit(f"device not found: {args.device}")

    for payload in payloads:
        raw = send_payload(args.device, args.baud, payload, timeout=args.timeout)
        print(json.dumps({"sent": payload, "tail_hex": raw[-128:].hex()}, ensure_ascii=False, indent=2))

    if args.pull_audio:
        print(json.dumps({"pulled": adb_pull_audio(args.pull_audio)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
