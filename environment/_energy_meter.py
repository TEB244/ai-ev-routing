"""
Lightweight in-process energy meter for inference runs on a workstation/lab
server (e.g. huron), where the DRAC portal can't resolve short jobs and
carbontracker/RAPL aren't available.

  GPU energy : MEASURED via NVML (pynvml) -- integrates real power.draw.
  CPU energy : ESTIMATED via a TDP model -- TDP * (process CPU fraction) * time,
               because no RAPL/powercap counter is exposed. Set the real chip TDP
               (e.g. 280 W for a Threadripper PRO 3975WX); the default 35.61 W
               fallbacks some tools use are an order of magnitude off.

Used as a context manager around the inference work:

    with EnergyMeter(cpu_tdp_w=280.0) as m:
        ... run inference ...
    print(m.total_kwh, m.gpu_kwh, m.cpu_kwh, m.seconds)

Robust by design: if pynvml/psutil are missing or error, that component reads 0
rather than crashing the run.
"""

import threading
import time


class EnergyMeter:
    def __init__(self, cpu_tdp_w=280.0, sample_hz=10.0,
                 carbon_intensity_g_per_kwh=170.0):
        self.cpu_tdp_w = float(cpu_tdp_w)
        self.interval = 1.0 / float(sample_hz)
        self.carbon_intensity = float(carbon_intensity_g_per_kwh)
        self._stop = threading.Event()
        self._samples = []          # list of (t, gpu_watts, cpu_fraction_of_chip)
        self._thread = None

        # GPU via NVML (measured).
        self._nvml = None
        self._handles = []
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            self._nvml = None

        # CPU via psutil (process utilization -> TDP estimate).
        self._proc = None
        self._ncpu = 1
        try:
            import os
            import psutil
            self._proc = psutil.Process()
            self._ncpu = os.cpu_count() or 1
            self._proc.cpu_percent(None)   # prime the counter
        except Exception:
            self._proc = None

    def _gpu_watts(self):
        if not self._nvml:
            return 0.0
        total = 0.0
        for h in self._handles:
            try:
                total += self._nvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW -> W
            except Exception:
                pass
        return total

    def _cpu_fraction(self):
        # Fraction of the whole chip our process is using (cores_used / total_cores).
        if not self._proc:
            return 0.0
        try:
            return min(1.0, (self._proc.cpu_percent(None) / 100.0) / self._ncpu)
        except Exception:
            return 0.0

    def _run(self):
        while not self._stop.is_set():
            self._samples.append((time.time(), self._gpu_watts(), self._cpu_fraction()))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.seconds = time.time() - self._t0

        gpu_j = cpu_j = 0.0
        s = self._samples
        for i in range(1, len(s)):
            dt = s[i][0] - s[i - 1][0]
            gpu_j += 0.5 * (s[i][1] + s[i - 1][1]) * dt
            cpu_w = self.cpu_tdp_w * 0.5 * (s[i][2] + s[i - 1][2])
            cpu_j += cpu_w * dt

        self.gpu_kwh = gpu_j / 3.6e6
        self.cpu_kwh = cpu_j / 3.6e6
        self.total_kwh = self.gpu_kwh + self.cpu_kwh
        self.co2_g = self.total_kwh * self.carbon_intensity
        self.n_samples = len(s)
        self.gpu_measured = self._nvml is not None
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        return False  # never suppress exceptions
