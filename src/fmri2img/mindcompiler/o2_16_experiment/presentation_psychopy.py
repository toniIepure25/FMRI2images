"""Real software-level PsychoPy presenter for the O2.16 experiment.

PsychoPy is imported LAZILY: this module always imports (so SIMULATION/CI keep working without PsychoPy), but
constructing/initialising a PsychoPyPresenter requires PsychoPy to be installed and pinned at the presentation
machine. Local timing measured through this presenter may be labelled only `PSYCHOPY_SOFTWARE_TIMING_VALIDATED`
-- never `MRI_DISPLAY_TIMING_VALIDATED`, which requires the actual MRI presentation display at the site.

Pinned target: PsychoPy == 2024.2.4 (record the exact installed version at runtime; the site freezes it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import presentation as _base
from . import timing as T

PINNED_PSYCHOPY_VERSION = "2024.2.4"


class PsychoPyUnavailable(RuntimeError):
    pass


def psychopy_available():
    try:
        import psychopy  # noqa: F401
        return True
    except Exception:
        return False


def psychopy_version():
    try:
        import psychopy
        return getattr(psychopy, "__version__", "unknown")
    except Exception:
        return None


@dataclass
class DisplayConfig:
    fullscreen: bool = True
    size: tuple = (1920, 1080)
    refresh_hz: Optional[float] = None       # TBD_SITE_OPERATOR until measured on the real display
    background_lum: float = 0.0
    units: str = "deg"
    screen: int = 0


class PsychoPyPresenter(_base.Presenter):
    """Implements the required interface: initialize_display / load_stimulus / draw_fixation /
    draw_visual_stimulus / draw_text_cue / flip / wait / capture_response / shutdown. All PsychoPy objects are
    created lazily so importing this module never requires PsychoPy."""

    def __init__(self, display: Optional[DisplayConfig] = None):
        self.display = display or DisplayConfig()
        self.win = None
        self._visual = None
        self._core = None
        self._event = None
        self._stimuli = {}
        self._frame_records = []

    # --- lifecycle ---
    def initialize_display(self):
        if not psychopy_available():
            raise PsychoPyUnavailable(
                "PsychoPy not installed. Install and pin PsychoPy==%s on the presentation machine; "
                "SIMULATION/CI use SimulationPresenter instead." % PINNED_PSYCHOPY_VERSION)
        from psychopy import visual, core, event
        self._visual, self._core, self._event = visual, core, event
        self.win = visual.Window(size=self.display.size, fullscr=self.display.fullscreen,
                                 color=[self.display.background_lum] * 3, units=self.display.units,
                                 screen=self.display.screen, waitBlanking=True)
        self.display.refresh_hz = self.win.getActualFrameRate() or self.display.refresh_hz
        return {"actual_refresh_hz": self.display.refresh_hz, "size": self.display.size}

    def load_stimulus(self, stim_id, image_path=None, text=None):
        if self.win is None:
            raise PsychoPyUnavailable("initialize_display() first")
        if image_path:
            self._stimuli[stim_id] = self._visual.ImageStim(self.win, image=image_path)
        else:
            self._stimuli[stim_id] = self._visual.TextStim(self.win, text=text or stim_id)
        return stim_id

    def draw_fixation(self):
        self._visual.TextStim(self.win, text="+").draw()

    def draw_visual_stimulus(self, stim_id):
        self._stimuli[stim_id].draw()

    def draw_text_cue(self, text):
        self._visual.TextStim(self.win, text=text).draw()

    def flip(self):
        return self.win.flip()             # returns the flip timestamp (PsychoPy clock)

    def wait(self, seconds):
        self._core.wait(seconds)

    def capture_response(self, keys, max_wait):
        pressed = self._event.waitKeys(maxWait=max_wait, keyList=list(keys), timeStamped=True)
        return pressed

    def shutdown(self):
        if self.win is not None:
            self.win.close(); self.win = None

    # --- base Presenter contract (used by task runners; requires an initialised window) ---
    def show(self, stimulus_ref, requested_onset_s, duration_s):
        if self.win is None:
            raise PsychoPyUnavailable("initialize_display() before show()")
        self.draw_text_cue(str(stimulus_ref))
        flip = self.flip()
        self.wait(duration_s)
        return flip, flip + duration_s, 0
