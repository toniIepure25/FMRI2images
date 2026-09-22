"""Local display characterization. Runs on ordinary hardware and reports PsychoPy software-timing quality. Its
result does NOT certify the MRI presentation display (that requires the site display + procedure). If PsychoPy
is absent it reports PSYCHOPY_UNAVAILABLE without failing."""
from __future__ import annotations

import statistics

from . import presentation_psychopy as PP


def characterize_display(n_frames=1000, size=(1280, 720), fullscreen=False):
    if not PP.psychopy_available():
        return {"status": "PSYCHOPY_UNAVAILABLE", "psychopy_version": None,
                "note": "Install PsychoPy==%s to characterize display locally. This never certifies MRI hardware." % PP.PINNED_PSYCHOPY_VERSION,
                "certifies_mri_hardware": False}
    from psychopy import visual, core
    win = visual.Window(size=size, fullscr=fullscreen, waitBlanking=True)
    detected_hz = win.getActualFrameRate(nIdentical=10, nMaxFrames=240) or None
    intervals = []
    last = core.getTime()
    stim = visual.TextStim(win, text="+")
    dropped = 0
    win.recordFrameIntervals = True
    for _ in range(n_frames):
        stim.draw(); win.flip()
        now = core.getTime(); intervals.append(now - last); last = now
    try:
        dropped = int(win.nDroppedFrames)
    except Exception:
        dropped = 0
    win.close()
    intervals = intervals[1:]  # drop first
    med = statistics.median(intervals) if intervals else None
    return {
        "status": "PSYCHOPY_SOFTWARE_TIMING_VALIDATED", "certifies_mri_hardware": False,
        "psychopy_version": PP.psychopy_version(), "requested_size": size, "fullscreen": fullscreen,
        "detected_refresh_hz": detected_hz, "n_frames": len(intervals),
        "median_frame_interval_s": med, "mean_frame_interval_s": (statistics.mean(intervals) if intervals else None),
        "frame_interval_sd_s": (statistics.pstdev(intervals) if len(intervals) > 1 else None),
        "max_frame_interval_s": (max(intervals) if intervals else None),
        "dropped_frames": dropped, "backend": "pyglet (PsychoPy default)",
        "label": "local software timing only; NOT MRI_DISPLAY_TIMING_VALIDATED",
    }
