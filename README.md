Check the index for the mic you are using in your pc using the below code


import pyaudio

def list_microphones():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')

    print("Available Microphones:")
    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            print(f"Device {i}: {device_info.get('name')}")

    p.terminate()

if __name__ == "__main__":
    list_microphones()
