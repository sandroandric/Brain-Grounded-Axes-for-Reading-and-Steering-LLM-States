# Timebase conventions

- `time_align` files are in **fMRI scan time** (audio starts 10.65 s after scan start). To get audio-onset time: `t_audio = t_align - 10.65`.
- MEG audio delivery delay: **+0.0395 s** from trigger to ear. If aligning MEG to audio, add 0.0395 s to stimulus times (or subtract from MEG times) consistently.
- Missing start triggers: sub-08_run-16 and sub-09_run-7. For these, `trig_start = trig_end - audio_duration`.
- Canonical timebases:
  - MEG sample time (seconds from MEG file start)
  - Audio-onset time (preferred semantic timebase)
  - fMRI scan time = audio time + 10.65 s
