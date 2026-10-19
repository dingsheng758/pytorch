# Owner(s): ["module: inductor"]
import contextlib
from unittest import mock

import torch
from torch._dynamo.device_interface import (
    CpuInterface,
    CudaInterface,
    DeviceInterface,
    MtiaInterface,
    MpsInterface,
    TpuInterface,
    XpuInterface,
)
from torch._inductor import config, cpp_builder
from torch._inductor.output_code import CompiledAOTI
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@contextlib.contextmanager
def _aoti_post_init_dispatch_env():
    """State under which CompiledAOTI.__post_init__ reaches the device
    dispatch: the autograd-for-AOTI gate is on and the host-OS early
    returns are neutralized so the tests run on any platform."""
    with (
        mock.patch.object(config, "enable_autograd_for_aot", True),
        mock.patch.object(cpp_builder, "_IS_WINDOWS", False),
        mock.patch.object(cpp_builder, "_IS_MACOS", False),
    ):
        yield


class TestAotiRunnerDeviceHook(TestCase):
    """Device-independent tests for DeviceInterface.create_aoti_runner().

    Runner constructors are mocked, so the hook wiring (base is abstract,
    per-backend overrides, exact ctor args, CompiledAOTI dispatch) is
    exercised on any hardware, CPU-only CI included.
    """
    hw_classification = HardwareClassification.GENERIC

    def test_base_hook_is_abstract(self):
        """Devices without an AOTI runtime must hit the base raise."""
        with self.assertRaises(NotImplementedError):
            DeviceInterface.create_aoti_runner("m.so", 1, "cpu")

    def test_aoti_backends_override_base_hook(self):
        for iface in (CpuInterface, CudaInterface, XpuInterface, MpsInterface):
            self.assertIsNot(
                iface.create_aoti_runner, DeviceInterface.create_aoti_runner
            )
        for iface in (MtiaInterface, TpuInterface):
            self.assertIs(
                iface.create_aoti_runner, DeviceInterface.create_aoti_runner
            )

    @parametrize(
        "iface,runner_attr,device_type,expected_args",
        [
            (CpuInterface, "AOTIModelContainerRunnerCpu", "cpu", ("m.so", 1)),
            (MpsInterface, "AOTIModelContainerRunnerMps", "mps", ("m.so", 1)),
            (
                XpuInterface,
                "AOTIModelContainerRunnerXpu",
                "xpu",
                ("m.so", 1, "xpu", ""),
            ),
            (
                CudaInterface,
                "AOTIModelContainerRunnerCuda",
                "cuda",
                ("m.so", 1, "cuda", "", True),
            ),
        ],
    )
    def test_hook_builds_runner_with_exact_args(
        self, iface, runner_attr, device_type, expected_args
    ):
        ctor = mock.MagicMock()
        ctor.return_value.run = mock.sentinel.run_fn
        with mock.patch.object(torch._C._aoti, runner_attr, ctor, create=True):
            run_fn = iface.create_aoti_runner("m.so", 1, device_type)
        self.assertIs(run_fn, mock.sentinel.run_fn)
        ctor.assert_called_once_with(*expected_args)

    def test_compiled_aoti_dispatches_through_device_interface(self):
        ctor = mock.MagicMock()
        ctor.return_value.run = mock.sentinel.run_fn
        with (
            _aoti_post_init_dispatch_env(),
            mock.patch.object(
                torch._C._aoti, "AOTIModelContainerRunnerCpu", ctor, create=True
            ),
        ):
            compiled = CompiledAOTI(filename="m.so", device_type="cpu")
        self.assertIs(compiled.current_callable, mock.sentinel.run_fn)
        self.assertTrue(compiled._boxed_call)
        ctor.assert_called_once_with("m.so", 1)

    def test_compiled_aoti_unsupported_device_raises(self):
        with _aoti_post_init_dispatch_env():
            with self.assertRaisesRegex(
                RuntimeError, "unsupported device type fake_device"
            ) as ctx:
                CompiledAOTI(filename="m.so", device_type="fake_device")
        self.assertIsInstance(ctx.exception.__cause__, NotImplementedError)

    def test_compiled_aoti_graph_module_skips_device_dispatch(self):
        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        ctor = mock.MagicMock()
        with (
            _aoti_post_init_dispatch_env(),
            mock.patch.object(
                torch._C._aoti, "AOTIModelContainerRunnerCpu", ctor, create=True
            ),
        ):
            compiled = CompiledAOTI(filename=gm, device_type="fake_device")
        self.assertIs(compiled.current_callable, gm)
        ctor.assert_not_called()


instantiate_parametrized_tests(TestAotiRunnerDeviceHook)


if __name__ == "__main__":
    run_tests()
