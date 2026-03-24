"""Screen manager for rendering device status and notifications to the OLED.

This module owns the ScreenWriter and a background thread that periodically
queries DeviceStatusManager to render a status dashboard. It also accepts
pushed notifications from other components for time-critical messages.

When camera preview is active, the manager takes ownership of the capture
device's preview context (via ``contextlib.ExitStack``) and renders live
dithered camera frames at the maximum rate the I2C bus allows.
"""
from typing import Optional

import collections
import contextlib
import dataclasses
import logging
import threading
import time

import numpy as np
import PIL.Image
import PIL.ImageChops
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageOps

from omgrab.cameras import cameras
from omgrab.devices import capture_device as capture_device_module
from omgrab.display import screen_writer as screen_writer_module
from omgrab.runtime import device_status as device_status_module

logger = logging.getLogger(__name__)

# How often to refresh the status screen (seconds).
_DEFAULT_REFRESH_INTERVAL_S = 1.0

# How long the screen stays on after the last activity (seconds).
_DEFAULT_SCREEN_TIMEOUT_S = 60.0


def _frame_to_display_image(
        frame: np.ndarray,
        display_w: int,
        display_h: int) -> PIL.Image.Image:
    """Convert an RGB frame to a 1-bit dithered image for the OLED.

    The frame is letterboxed (black bars added) to fit the display
    aspect ratio without cropping, so the entire field of view is
    visible. Based on the reference implementation in
    ``omgrab.display.test_alignment``.

    Args:
        frame: RGB numpy array (H, W, 3).
        display_w: Target display width in pixels.
        display_h: Target display height in pixels.

    Returns:
        1-bit PIL Image ready for the display.
    """
    image = PIL.Image.fromarray(frame)
    src_w, src_h = image.size

    target_ratio = display_w / display_h
    src_ratio = src_w / src_h

    # Equalize before letterboxing so the black padding pixels don't
    # pollute the histogram (which would reduce contrast and cause
    # dithering artifacts at the image/bar boundary).
    image = image.convert('L')
    image = PIL.ImageOps.equalize(image)

    if abs(src_ratio - target_ratio) < 0.01:
        # Aspect ratios match -- just resize.
        image = image.resize((display_w, display_h),
                             PIL.Image.Resampling.LANCZOS)
    elif src_ratio > target_ratio:
        # Source is wider -- fit to width, letterbox top/bottom.
        new_w = display_w
        new_h = int(display_w / src_ratio)
        image = image.resize((new_w, new_h),
                             PIL.Image.Resampling.LANCZOS)
        pad_top = (display_h - new_h) // 2
        padded = PIL.Image.new('L', (display_w, display_h), 0)
        padded.paste(image, (0, pad_top))
        image = padded
    else:
        # Source is taller -- fit to height, pillarbox left/right.
        new_h = display_h
        new_w = int(display_h * src_ratio)
        image = image.resize((new_w, new_h),
                             PIL.Image.Resampling.LANCZOS)
        pad_left = (display_w - new_w) // 2
        padded = PIL.Image.new('L', (display_w, display_h), 0)
        padded.paste(image, (pad_left, 0))
        image = padded

    # Floyd-Steinberg dither to 1-bit.
    image = image.convert('1')
    return image


@dataclasses.dataclass
class Notification:
    """A transient notification to display on screen.

    Attributes:
        message: Text to display.  For warnings, use a newline
            to separate into two lines.
        duration_s: How long to show the notification.
        priority: Higher values are shown first.
        warning: If True, render with a triangle alert symbol.
        created_at: Monotonic timestamp when the notification was created.
    """

    message: str
    duration_s: float = 3.0
    priority: int = 0
    warning: bool = False
    created_at: float = dataclasses.field(default_factory=time.monotonic)


class ScreenManager:
    """Manages the OLED display with periodic status updates and push notifications.

    The manager runs a background thread that:
    1. Checks for pending notifications (highest priority first).
    2. If no notification is active, renders the current device status.
    3. Pushes the rendered image to the ScreenWriter.

    Other components can call show_notification() from any thread to display
    a transient message immediately.
    """

    def __init__(self,
                 writer: Optional[screen_writer_module.ScreenWriter] = None,
                 refresh_interval_s: float = _DEFAULT_REFRESH_INTERVAL_S,
                 screen_timeout_s: float = _DEFAULT_SCREEN_TIMEOUT_S):
        """Initialize the screen manager.

        Args:
            writer: ScreenWriter instance. If None, a default one is created.
            refresh_interval_s: How often to refresh the display (seconds).
            screen_timeout_s: Seconds of inactivity before blanking the screen.
        """
        self._writer = writer or screen_writer_module.ScreenWriter()
        self._refresh_interval_s = refresh_interval_s
        self._screen_timeout_s = screen_timeout_s

        # Component reference (set after construction via setter).
        self._device_status_manager: Optional[
            device_status_module.DeviceStatusManager] = None

        # Fonts for rendering text.
        self._font_small = PIL.ImageFont.load_default(size=12)
        self._font_large = PIL.ImageFont.load_default(size=18)

        # Notification queue (thread-safe via lock).
        self._notifications: collections.deque[Notification] = collections.deque()
        self._notifications_lock = threading.Lock()
        self._active_notification: Optional[Notification] = None

        # Screen timeout tracking.
        self._last_activity_time: float = time.monotonic()

        # Alert cooldowns: alert_key -> monotonic time when last shown.
        self._alert_last_shown: dict[str, float] = {}

        # Whether the screen is currently blanked (for logging transitions).
        self._screen_asleep = False

        # Camera preview state.
        self._capture_device: Optional[
            capture_device_module.CaptureDevice] = None
        self._preview_camera: Optional[cameras.Camera] = None
        self._preview_active = False
        self._preview_stack: Optional[contextlib.ExitStack] = None

        # Worker thread.
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def set_preview_source(
            self,
            capture_device: capture_device_module.CaptureDevice,
            preview_camera: cameras.Camera):
        """Set the capture device and preview camera for camera preview mode.

        Args:
            capture_device: The capture device that provides the preview
                context manager.
            preview_camera: The camera to read preview frames from.
        """
        self._capture_device = capture_device
        self._preview_camera = preview_camera

    def set_device_status_manager(
            self,
            device_status_manager: device_status_module.DeviceStatusManager):
        """Set the device status manager reference.

        Args:
            device_status_manager: The DeviceStatusManager to query for status.
        """
        self._device_status_manager = device_status_manager

    def start(self):
        """Start the background display refresh thread."""
        if self._worker_thread is not None:
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name='screen-manager',
        )
        self._worker_thread.start()
        logger.info('Screen manager started (interval=%.1fs, available=%s)',
                     self._refresh_interval_s, self._writer.available)

    def shutdown(self):
        """Stop the background thread and clean up the display."""
        # Stop preview if active (closes the device preview context).
        if self._preview_active:
            self.stop_preview()

        self._stop_event.set()
        self._wake_event.set()  # Unblock any wait.
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3.0)
            self._worker_thread = None

        # Show a shutdown message before turning off the display.
        try:
            width = self._writer.width
            height = self._writer.height
            image = PIL.Image.new('1', (width, height), 0)
            draw = PIL.ImageDraw.Draw(image)
            self._draw_centered_body(draw, width, 'Shutting', 'down...')
            self._writer.display(image)
        except Exception as e:
            logger.debug('Could not show shutdown screen: %s', e)

        self._writer.cleanup()
        logger.info('Screen manager stopped')

    @property
    def display_available(self) -> bool:
        """Whether the OLED display hardware is currently available."""
        return self._writer.available

    def start_preview(self):
        """Start live camera preview on the display.

        Enters the capture device's preview context (starting the
        lightweight RGB-only pipeline) and switches the render loop to
        display dithered camera frames at the maximum rate the I2C bus
        allows.

        Raises:
            RuntimeError: If the preview source has not been set, the
                device cannot start preview, or the display is unavailable.
        """
        if self._capture_device is None or self._preview_camera is None:
            raise RuntimeError('Preview source not set (call set_preview_source first)')
        if not self._writer.available:
            raise RuntimeError('Display hardware not available')

        self._preview_stack = contextlib.ExitStack()
        self._preview_stack.enter_context(self._capture_device.preview())
        self._preview_active = True
        self.wake()
        logger.info('Camera preview started')

    def stop_preview(self):
        """Stop live camera preview and return to normal status display."""
        self._preview_active = False
        if self._preview_stack is not None:
            self._preview_stack.close()
            self._preview_stack = None
        self.wake()
        logger.info('Camera preview stopped')

    def wake(self):
        """Reset the screen timeout and wake the display immediately.

        Call from any thread (e.g. on button press) to turn the screen
        back on if it has timed out.
        """
        if self._screen_asleep:
            logger.info('Screen waking up')
            self._screen_asleep = False
        self._last_activity_time = time.monotonic()
        self._wake_event.set()

    def show_notification(self, message: str, duration_s: float = 3.0,
                          priority: int = 0, warning: bool = False):
        """Push a transient notification to the display.

        Can be called from any thread. The notification will be shown
        immediately (waking the display thread) and auto-dismissed after
        duration_s seconds.

        Args:
            message: Text to display.  For warnings, use a newline
                to separate into two lines.
            duration_s: How long to show the notification (seconds).
            priority: Higher values are shown first when multiple are queued.
            warning: If True, render with a triangle alert symbol.
        """
        notification = Notification(
            message=message,
            duration_s=duration_s,
            priority=priority,
            warning=warning,
        )
        with self._notifications_lock:
            self._notifications.append(notification)
        # Wake the worker immediately (also resets timeout).
        self.wake()

    def _worker_loop(self):
        """Background loop: render notifications or status, push to display."""
        while not self._stop_event.is_set():
            try:
                image = self._render_frame()
                self._writer.display(image)
            except Exception as e:
                logger.warning('Error in screen manager loop: %s', e)

            if not self._preview_active:
                # Normal mode: refresh at configured interval.
                self._wake_event.wait(timeout=self._refresh_interval_s)
                self._wake_event.clear()
            # In preview mode: loop immediately for max frame rate.

    def _render_frame(self) -> PIL.Image.Image:
        """Decide what to render and return the image.

        The screen blanks after a period of inactivity to prevent OLED
        burn-in. Recording, uploading, preview, and notifications count
        as activity and keep the screen on.

        Returns:
            A 1-bit PIL Image ready for the display.
        """
        # Camera preview mode: render live camera frames.
        if self._preview_active:
            self._last_activity_time = time.monotonic()
            return self._render_preview_frame()

        now = time.monotonic()

        # Check if there's an active notification still being shown.
        if self._active_notification is not None:
            elapsed = now - self._active_notification.created_at
            if elapsed < self._active_notification.duration_s:
                return self._render_notification(self._active_notification)
            # Notification expired.
            self._active_notification = None

        # Check for new notifications (pick highest priority).
        notification = self._pop_next_notification()
        if notification is not None:
            self._active_notification = notification
            return self._render_notification(notification)

        # Check whether the device is busy (keeps the screen on).
        status = self._get_device_status()
        if status is not None:
            is_busy = status.recording.is_recording
            if is_busy:
                self._last_activity_time = now

            # Generate threshold-based alerts (battery, storage).
            self._check_alerts(status, now)

        # If an alert was just queued, show it immediately.
        notification = self._pop_next_notification()
        if notification is not None:
            self._active_notification = notification
            return self._render_notification(notification)

        # Blank the screen after the timeout when idle.
        if now - self._last_activity_time > self._screen_timeout_s:
            if not self._screen_asleep:
                logger.info('Screen sleeping after %.0fs of inactivity',
                            self._screen_timeout_s)
                self._screen_asleep = True
            return PIL.Image.new('1', (self._writer.width,
                                       self._writer.height), 0)

        # Default: render status screen (pass pre-fetched status).
        return self._render_status_screen(status=status)

    def _check_alerts(self, status: device_status_module.DeviceStatus,
                       now: float):
        """Generate threshold-based alerts from device status.

        Checks battery and storage levels and queues warning
        notifications when thresholds are exceeded.  Each alert type
        has an independent cooldown to avoid spamming.

        Args:
            status: Current device status snapshot.
            now: Current monotonic time.
        """
        # Low battery
        if status.battery is not None:
            pct = status.battery.percent
            if pct <= 5:
                self._maybe_show_alert(
                    'battery_critical', now,
                    f'Low Battery\n{int(pct)}%',
                    duration_s=3600.0, cooldown_s=5.0, priority=20,
                    warning=True,
                )
            elif pct <= 10:
                self._maybe_show_alert(
                    'battery_low', now,
                    f'Low Battery\n{int(pct)}%',
                    duration_s=5.0, cooldown_s=5.0, priority=10,
                    warning=True,
                )

        # Storage almost full
        if status.storage.used_percent >= 90:
            self._maybe_show_alert(
                'storage_full', now,
                'Storage\nAlmost Full',
                duration_s=5.0, cooldown_s=5.0, priority=10,
                warning=True,
            )

    def _maybe_show_alert(self, key: str, now: float, message: str,
                           duration_s: float, cooldown_s: float,
                           priority: int, warning: bool):
        """Show an alert if it hasn't been shown recently.

        Skips the cooldown check when another notification is currently
        active (so alerts don't get suppressed while an unrelated
        notification is on screen).

        Args:
            key: Unique alert identifier for cooldown tracking.
            now: Current monotonic time.
            message: Alert message text.
            duration_s: How long to display the alert.
            cooldown_s: Minimum interval between showings.
            priority: Notification priority.
            warning: Whether to render as a warning (triangle symbol).
        """
        last = self._alert_last_shown.get(key, 0.0)
        # Don't suppress if another notification is active (cooldown
        # only applies when nothing else is occupying the screen).
        if self._active_notification is None and now - last < cooldown_s + duration_s:
                return

        self._alert_last_shown[key] = now
        self.show_notification(
            message, duration_s=duration_s, priority=priority,
            warning=warning,
        )

    def _pop_next_notification(self) -> Optional[Notification]:
        """Pop the highest-priority notification from the queue.

        Discards any notifications that have already expired.

        Returns:
            The next notification to display, or None if the queue is empty.
        """
        now = time.monotonic()
        with self._notifications_lock:
            # Remove expired notifications.
            self._notifications = collections.deque(
                n for n in self._notifications
                if now - n.created_at < n.duration_s
            )
            if not self._notifications:
                return None

            # Sort by priority (highest first) and pop.
            best = max(self._notifications, key=lambda n: n.priority)
            self._notifications.remove(best)
            return best

    def _render_notification(self,
                             notification: Notification) -> PIL.Image.Image:
        """Render a notification message to an image.

        Warning notifications show a triangle alert symbol at the top
        with up to two lines of text below.  Info notifications use
        a border with up to three word-wrapped lines.

        Args:
            notification: The notification to render.

        Returns:
            A 1-bit PIL Image with the notification text.
        """
        width = self._writer.width
        height = self._writer.height
        image = PIL.Image.new('1', (width, height), 0)
        draw = PIL.ImageDraw.Draw(image)

        if notification.warning:
            return self._render_warning(image, draw, notification.message)

        # -- Info notification (border + word-wrapped text) --
        padding = 8
        draw.rectangle([0, 0, width - 1, height - 1], outline=255)

        # Word-wrap the message within the display width (with padding).
        lines = self._word_wrap(
            notification.message, draw,
            self._font_large, width - padding * 2)  # type: ignore[arg-type]

        # Center vertically.
        line_height = 20
        total_text_height = len(lines) * line_height
        y_start = max(padding, (height - total_text_height) // 2)

        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=self._font_large)
            lw = bbox[2] - bbox[0]
            x = (width - lw) // 2
            draw.text((x, y_start + i * line_height), line,
                      font=self._font_large, fill=255)

        return image

    def _render_warning(self, image: PIL.Image.Image,
                        draw: PIL.ImageDraw.ImageDraw,
                        message: str) -> PIL.Image.Image:
        """Render a warning with a triangle symbol and up to two text lines.

        Layout:
            - Filled triangle with '!' cutout, centred at top.
            - Up to two lines of large text below, centred.

        Args:
            image: The display image.
            draw: ImageDraw instance.
            message: Warning text; use newline to split into two lines.

        Returns:
            The rendered image.
        """
        width = self._writer.width

        # -- Triangle alert symbol (centred, 18px tall) --
        tri_h = 18
        tri_w = 20
        cx = width // 2
        tri_top = 2
        tri = [
            (cx, tri_top),                              # apex
            (cx - tri_w // 2, tri_top + tri_h),         # bottom-left
            (cx + tri_w // 2, tri_top + tri_h),         # bottom-right
        ]
        draw.polygon(tri, fill=255)

        # '!' cutout via black text over the white triangle.
        bang_bbox = draw.textbbox((0, 0), '!', font=self._font_small)
        bang_w = bang_bbox[2] - bang_bbox[0]
        bang_h = bang_bbox[3] - bang_bbox[1]
        bang_x = cx - bang_w // 2
        bang_y = tri_top + (tri_h - bang_h) // 2 + 1  # nudge down a bit
        draw.text((bang_x, bang_y), '!', font=self._font_small, fill=0)

        # -- Text below the triangle (up to 2 lines, centred) --
        lines = message.split('\n', maxsplit=1)
        font = self._font_large
        text_top = tri_top + tri_h + 4
        line_spacing = 4

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(((width - tw) // 2, text_top), line,
                      font=font, fill=255)
            text_top += int(bbox[3] - bbox[1]) + line_spacing

        return image

    def _render_preview_frame(self) -> PIL.Image.Image:
        """Render a single live camera preview frame.

        Grabs the freshest frame from the preview camera, converts it to
        a 1-bit dithered image, and returns it. If no frame is available
        within the timeout, returns a blank (black) image.

        Returns:
            A 1-bit PIL Image ready for the display.
        """
        width = self._writer.width
        height = self._writer.height
        blank = PIL.Image.new('1', (width, height), 0)

        if self._preview_camera is None:
            return blank

        # Get a frame (blocks briefly if the queue is empty).
        try:
            frame, _ts = self._preview_camera.get_next_frame(timeout_s=0.5)
        except cameras.FrameUnavailableError:
            return blank

        # Drain older frames so we always display the freshest one.
        while True:
            try:
                frame, _ts = self._preview_camera.get_next_frame(timeout_s=0)
            except cameras.FrameUnavailableError:
                break

        return _frame_to_display_image(frame, width, height)

    def _render_status_screen(
            self,
            status: Optional[device_status_module.DeviceStatus] = None) -> PIL.Image.Image:
        """Render the device status dashboard to an image.

        Layout (128x64):
            Top:    [WiFi bars]              [Battery bars]

            When recording (two centered lines, large font):
                "Recording"
                "XXmYYs"

            When idle (two centered lines, large font):
                "Ready"
                "Uploading..." or "All clips uploaded"

        Args:
            status: Pre-fetched status snapshot, or None to fetch fresh.

        Returns:
            A 1-bit PIL Image with the status dashboard.
        """
        width = self._writer.width
        height = self._writer.height
        image = PIL.Image.new('1', (width, height), 0)
        draw = PIL.ImageDraw.Draw(image)

        if status is None:
            status = self._get_device_status()
        if status is None:
            self._draw_centered_body(draw, width, 'Starting...', '')
            return image

        # Top row (y=0..15): WiFi icon left, battery icon right
        wifi_signal = status.network.wifi_signal_strength
        wifi_connected = status.network.wifi_ssid is not None
        wifi_no_internet = (wifi_connected
                            and status.network.status in ('offline',
                                                          'network_only'))
        self._draw_wifi_icon(image, draw, x=2, y=1, connected=wifi_connected,
                             signal_dbm=wifi_signal,
                             no_internet=wifi_no_internet)

        batt_percent = status.battery.percent if status.battery else None
        batt_charging = (status.battery is not None
                         and status.battery.current_a > 0)
        self._draw_battery_icon(image, draw, x=width - 28, y=2,
                                percent=batt_percent, charging=batt_charging)

        # Body (y=18..63): two centered lines in the large font
        if status.recording.is_recording:
            line1 = 'Recording'
            if status.recording.duration_seconds is not None:
                total_secs = int(status.recording.duration_seconds)
                hours = total_secs // 3600
                mins = (total_secs % 3600) // 60
                secs = total_secs % 60
                if hours > 0:
                    line2 = f'{hours:02d}h{mins:02d}m{secs:02d}s'
                else:
                    line2 = f'{mins:02d}m{secs:02d}s'
            else:
                line2 = ''
        else:
            line1 = 'Ready'
            line2 = ''

        self._draw_centered_body(draw, width, line1, line2)

        return image

    def _draw_centered_body(self, draw: PIL.ImageDraw.ImageDraw,
                            width: int, line1: str, line2: str):
        """Draw one or two lines of large text centred in the body area.

        The body area spans y=18..63 (46px). Lines are vertically centred
        within this region.

        Args:
            draw: ImageDraw instance.
            width: Display width in pixels.
            line1: Primary text line.
            line2: Secondary text line (may be empty).
        """
        body_top = 18
        body_height = 46
        line_spacing = 4
        font = self._font_large

        bbox1 = draw.textbbox((0, 0), line1, font=font)
        h1 = bbox1[3] - bbox1[1]

        if line2:
            bbox2 = draw.textbbox((0, 0), line2, font=font)
            h2 = bbox2[3] - bbox2[1]
            total_h = h1 + line_spacing + h2
            y1 = body_top + (body_height - total_h) // 2
            y2 = y1 + h1 + line_spacing

            w1 = bbox1[2] - bbox1[0]
            draw.text(((width - w1) // 2, y1), line1, font=font, fill=255)

            w2 = bbox2[2] - bbox2[0]
            draw.text(((width - w2) // 2, y2), line2, font=font, fill=255)
        else:
            y1 = body_top + (body_height - h1) // 2
            w1 = bbox1[2] - bbox1[0]
            draw.text(((width - w1) // 2, y1), line1, font=font, fill=255)

    def _draw_wifi_icon(self, image: PIL.Image.Image,
                        draw: PIL.ImageDraw.ImageDraw,
                        x: int, y: int, connected: bool,
                        signal_dbm: Optional[int],
                        no_internet: bool = False):
        """Draw a classic WiFi fan icon (concentric arcs + dot).

        Three arcs represent signal strength, with a dot at the bottom
        centre as the base. Arcs are drawn or left empty based on dBm.

        Args:
            image: The display image (needed for XOR '!' rendering).
            draw: ImageDraw instance.
            x: Left edge of the icon.
            y: Top edge of the icon.
            connected: Whether WiFi is connected at all.
            signal_dbm: Signal strength in dBm, or None.
            no_internet: If True, draw a '!' over the icon via XOR to
                indicate WiFi is connected but internet is unreachable.
        """
        # Determine how many arcs to fill (0-3).
        if not connected or signal_dbm is None:
            arcs = 0
        elif signal_dbm >= -50:
            arcs = 3
        elif signal_dbm >= -65:
            arcs = 2
        else:
            arcs = 1

        # Icon dimensions: 16px wide, 14px tall.
        # Centre-bottom of the fan is the dot.
        cx = x + 8
        cy = y + 13

        # Draw the base dot.
        dot_r = 1
        draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                     fill=255)

        # Draw concentric arcs (inner to outer).
        # When disconnected, draw all arcs fully to show the "full stack"
        # shape, then strike through with a diagonal line.
        arc_radii = [5, 9, 13]
        for i, r in enumerate(arc_radii):
            bbox = [cx - r, cy - r, cx + r, cy + r]
            if not connected:
                # Full stack outline for disconnected icon.
                draw.arc(bbox, start=225, end=315, fill=255, width=2)
            elif i < arcs:
                draw.arc(bbox, start=225, end=315, fill=255, width=2)
            else:
                # Connected but weak: show thin outline for unfilled arcs.
                draw.arc(bbox, start=225, end=315, fill=255, width=1)

        # Diagonal strike-through when not connected (bottom-left to top-right).
        if not connected:
            draw.line([x, cy, x + 16, y], fill=255, width=2)

        if no_internet:
            # Draw a white ! to the right of the wifi icon
            draw.text((cx + 14, y - 4), '!', font=self._font_large, fill=255)

    def _draw_battery_icon(self, image: PIL.Image.Image,
                           draw: PIL.ImageDraw.ImageDraw,
                           x: int, y: int, percent: Optional[float],
                           charging: bool = False):
        """Draw a battery icon with fill level and optional charging bolt.

        The charging bolt is drawn via XOR so that each pixel is always the
        inverse of whatever is behind it (white on empty, black on fill),
        ensuring visibility at any charge level.

        Args:
            image: The display image (needed for XOR bolt rendering).
            draw: ImageDraw instance.
            x: Left edge of the icon.
            y: Top edge of the icon.
            percent: Battery percentage (0-100), or None if unavailable.
            charging: Whether the battery is currently charging.
        """
        # Battery body: 24x10 rectangle.
        body_w = 24
        body_h = 10
        draw.rectangle([x, y, x + body_w - 1, y + body_h - 1], outline=255)

        # Battery tip (positive terminal nub): 2x4 on the right.
        tip_h = 4
        tip_y = y + (body_h - tip_h) // 2
        draw.rectangle([x + body_w, tip_y, x + body_w + 1, tip_y + tip_h - 1],
                       fill=255)

        if percent is None:
            # Draw a ? in the middle to indicate unknown.
            draw.text((x + 7, y - 1), '?', font=self._font_small, fill=255)
            return

        # Fill proportional to percentage (inside the outline with 1px pad).
        fill_max_w = body_w - 4  # 2px padding each side.
        fill_w = max(0, int(fill_max_w * percent / 100.0))
        if fill_w > 0:
            draw.rectangle([x + 2, y + 2, x + 2 + fill_w - 1,
                            y + body_h - 3], fill=255)

        # Lightning bolt overlay when charging, drawn via XOR.
        if charging:
            cx = x + body_w // 2  # Centre of the battery body.
            cy = y + body_h // 2
            bolt = [
                (cx + 1, y),          # top
                (cx - 3, cy),         # left-middle
                (cx, cy),             # centre
                (cx - 2, y + body_h), # bottom
                (cx + 3, cy),         # right-middle
                (cx, cy),             # centre
            ]
            # Draw bolt onto a mask, then XOR with the image so each bolt
            # pixel is the inverse of whatever is behind it.
            bolt_mask = PIL.Image.new('1', image.size, 0)
            PIL.ImageDraw.Draw(bolt_mask).polygon(bolt, fill=255)
            image.paste(PIL.ImageChops.logical_xor(image, bolt_mask))

    def _get_device_status(
            self) -> Optional[device_status_module.DeviceStatus]:
        """Get the current device status snapshot.

        Returns:
            DeviceStatus if the manager is wired, else None.
        """
        if self._device_status_manager is None:
            return None
        try:
            return self._device_status_manager.get_status()
        except Exception as e:
            logger.debug('Failed to get device status for display: %s', e)
            return None

    @staticmethod
    def _word_wrap(text: str, draw: PIL.ImageDraw.ImageDraw,
                   font: PIL.ImageFont.ImageFont,
                   max_width: int) -> list[str]:
        """Wrap text to fit within a pixel width.

        Args:
            text: The text to wrap.
            draw: ImageDraw instance for measurement.
            font: Font to measure with.
            max_width: Maximum line width in pixels.

        Returns:
            List of wrapped lines.
        """
        words = text.split()
        lines: list[str] = []
        current_line = ''

        for word in words:
            test_line = f'{current_line} {word}'.strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if (bbox[2] - bbox[0]) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines or ['']
