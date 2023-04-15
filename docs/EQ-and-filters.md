## Cakewalk peak filters

This type of filter is activated by `fil_type=bpk_2p` (not `pkf_2p`).
The peak gain is controlled by `resonance`, not `fil_gain` which is ARIA only.

The peak is highest at `resonance=0` and attenuates for higher values.
It is not yet studied how the `resonance` control and peak gain are related.