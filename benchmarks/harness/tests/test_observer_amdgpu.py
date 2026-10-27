import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import observer  # noqa: E402


def make_card(root, name, vendor, **files):
    """A fake /sys/class/drm/<name>/device tree; files maps relative path -> text."""
    device = Path(root) / name / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text(vendor + "\n")
    for rel, text in files.items():
        path = device / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(text) + "\n")
    return device


FULL = {
    "mem_info_vram_used": 500,
    "mem_info_gtt_used": 4000,
    "gpu_busy_percent": 37,
    "hwmon/hwmon3/power1_average": 25_500_000,
    "hwmon/hwmon3/power1_input": 99_000_000,
}


class AmdgpuDeviceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_reads_memory_sum_gtt_util_and_average_power(self):
        make_card(self.root, "card1", "0x1002", **FULL)
        dev = observer._amdgpu_device(0, self.root)
        self.assertEqual(dev.used_bytes(), 4500)
        self.assertEqual(dev.gtt_used_bytes(), 4000)
        self.assertEqual(dev.util_pct(), 37)
        self.assertEqual(dev.power_w(), 25.5)

    def test_power_falls_back_to_input_when_average_is_absent(self):
        files = {k: v for k, v in FULL.items() if not k.endswith("power1_average")}
        make_card(self.root, "card0", "0x1002", **files)
        self.assertEqual(observer._amdgpu_device(0, self.root).power_w(), 99.0)

    def test_picks_amd_card_skipping_other_vendors_and_connectors(self):
        make_card(self.root, "card0", "0x8086", **FULL)  # Intel
        make_card(self.root, "card1", "0x1002", **{**FULL, "mem_info_gtt_used": 7})
        make_card(self.root, "card1-DP-1", "0x1002", **FULL)  # a connector, not a card
        make_card(self.root, "card2", "0x1002", **{**FULL, "mem_info_gtt_used": 9})
        self.assertEqual(observer._amdgpu_device(0, self.root).gtt_used_bytes(), 7)
        self.assertEqual(observer._amdgpu_device(1, self.root).gtt_used_bytes(), 9)
        self.assertIsNone(observer._amdgpu_device(2, self.root))

    def test_missing_files_give_none_not_a_crash(self):
        make_card(self.root, "card0", "0x1002")
        dev = observer._amdgpu_device(0, self.root)
        self.assertIsNone(dev.used_bytes())
        self.assertIsNone(dev.gtt_used_bytes())
        self.assertIsNone(dev.util_pct())
        self.assertIsNone(dev.power_w())

    def test_memory_needs_both_pools(self):
        make_card(self.root, "card0", "0x1002", mem_info_gtt_used=5)
        self.assertIsNone(observer._amdgpu_device(0, self.root).used_bytes())

    def test_unparsable_file_is_none(self):
        make_card(self.root, "card0", "0x1002", gpu_busy_percent="n/a")
        self.assertIsNone(observer._amdgpu_device(0, self.root).util_pct())

    def test_no_drm_root_or_no_amd_card(self):
        self.assertIsNone(observer._amdgpu_device(0, self.root / "absent"))
        make_card(self.root, "card0", "0x10de", **FULL)
        self.assertIsNone(observer._amdgpu_device(0, self.root))

    def test_module_functions_dispatch_on_the_handle(self):
        make_card(self.root, "card0", "0x1002", **FULL)
        dev = observer._amdgpu_device(0, self.root)
        self.assertEqual(observer._device_used_bytes(dev), 4500)
        self.assertEqual(observer._device_gtt_bytes(dev), 4000)
        self.assertEqual(observer._device_util_pct(dev), 37)
        self.assertEqual(observer._device_power_w(dev), 25.5)
        self.assertIsNone(observer._device_gtt_bytes(None))


class SamplerSelectionTests(unittest.TestCase):
    def test_amd_names_select_amdgpu(self):
        for name in ("AMD Radeon Graphics", "Radeon 8060S", "AMD Instinct MI300X"):
            self.assertTrue(observer._is_amd_device(name), name)

    def test_nvidia_and_unreported_do_not(self):
        for name in ("NVIDIA GeForce RTX 5090", "NVIDIA H200", None):
            self.assertFalse(observer._is_amd_device(name), name)

    def test_gpu_handle_routes_by_device(self):
        sentinel = object()
        with unittest.mock.patch.object(observer, "_nvml_handle", return_value=sentinel) as nvml, \
                unittest.mock.patch.object(observer, "_amdgpu_device", return_value="amd") as amd:
            self.assertIs(observer._gpu_handle("NVIDIA H200", 0), sentinel)
            self.assertEqual(observer._gpu_handle("AMD Radeon Graphics", 1), "amd")
            nvml.assert_called_once_with(0)
            amd.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
