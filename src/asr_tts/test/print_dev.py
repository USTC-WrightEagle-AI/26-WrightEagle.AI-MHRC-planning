import sounddevice as sd

devices = sd.query_devices()
for i, device in enumerate(devices):
    print(f"{i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})")