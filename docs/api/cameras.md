# cameras

Camera classes represent a particular capture session or recording duration.
A camera is typically created from a capture device, though cameras can also
operate independently (e.g. opening a USB device directly via OpenCV).
Cameras are started and stopped each time the user begins and ends a
recording, while the underlying capture device may remain powered on across
multiple recordings.

::: omgrab.cameras.cameras

::: omgrab.cameras.oakd_camera

::: omgrab.cameras.queue_reader_camera

::: omgrab.cameras.usb_camera
