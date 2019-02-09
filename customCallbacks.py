from rl.callbacks import Callback
import json
import numpy as np
from subprocess import Popen, PIPE, STDOUT
import psutil as ps


def gpu_mon():
    cmd = "nvidia-smi --format=csv,noheader,nounits --query-gpu=fan.speed,memory.total,memory.used,memory.free," \
          "utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit"
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output = p.stdout.read().decode('ASCII')
    output = output[:len(output) - 3]
    values = output.split(sep=",")
    real_values = {
        "fan_speed": int(values[0]),
        "mem_used": float(values[2]),
        "gpu_util": int(values[4]),
        "mem_util": int(values[5]),
        "gpu_temp": int(values[6]),
        "gpu_power": float(values[7])
    }

    cpu = ps.cpu_percent()
    ram = dict(ps.virtual_memory()._asdict())
    swap = dict(ps.swap_memory()._asdict())

    real_values["cpu_util"] = cpu
    real_values["ram_util"] = ram["percent"]
    real_values["swap_util"] = swap["percent"]

    return real_values


class GpuLogger(Callback):
    def __init__(self, filepath, interval=None, printing=False):
        self.filepath = filepath
        self.interval = interval
        self.printing = printing

        self.data = {}

    def on_train_end(self, logs=None):
        """ Save model at the end of training """
        self.save_data()

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""
        data = gpu_mon()
        data["episode"] = episode

        for key, value in data.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

        if self.interval is not None and episode % self.interval == 0:
            self.save_data()
            if self.printing:
                print(data)

    def save_data(self):
        """ Save metrics in a json file """
        if len(self.data.keys()) == 0:
            return

        # Sort everything by episode.
        assert 'episode' in self.data
        sorted_indexes = np.argsort(self.data['episode'])
        sorted_data = {}
        for key, values in self.data.items():
            assert len(self.data[key]) == len(sorted_indexes)
            # We convert to np.array() and then to list to convert from np datatypes to native datatypes.
            # This is necessary because json.dump cannot handle np.float32, for example.
            sorted_data[key] = np.array([self.data[key][idx] for idx in sorted_indexes]).tolist()

        # Overwrite already open file. We can simply seek to the beginning since the file will
        # grow strictly monotonously.
        with open(self.filepath, 'w') as f:
            json.dump(sorted_data, f)
