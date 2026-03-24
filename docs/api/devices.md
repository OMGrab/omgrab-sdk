# devices

Capture devices represent the physical hardware (or driver/pipeline) that
produces frame data. A device owns the underlying resources (USB handles,
DepthAI pipelines, etc.) and can be powered on or off independently of any
recording. Cameras, by contrast, are lightweight per-recording wrappers
typically created from a capture device.

For example, the OAK-D capture device manages the DepthAI pipeline and
hardware connection, while `OakDRGBCamera` and `OakDDepthCamera` are
session-scoped readers created for each recording. The device may stay
powered between recordings (e.g. for preview), but cameras are created
fresh each time.

::: omgrab.devices.capture_device

::: omgrab.devices.oakd_capture_device

::: omgrab.devices.usb_capture_device
