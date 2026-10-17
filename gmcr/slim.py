"""Copy a gmprocess workspace keeping only what gmcr reads (about 36% of the size).

Dropped: the full-resolution Signal/Event/Noise spectrum caches and the cached
pre-event noise trace (the smoothed spectra and SNR curves are kept). Features
extracted from a slim copy are identical to those from the original.
"""

from pathlib import Path

import h5py

DROP_CACHE = {
    "SignalSpectrumSpec", "SignalSpectrumFreq",
    "EventSpectrumSpec", "EventSpectrumFreq",
    "NoiseSpectrumSpec", "NoiseSpectrumFreq",
    "PreeventNoiseTraceData", "PreeventNoiseTraceTimes",
}


def slim(src, dst):
    """Write the slim copy of src to dst (atomically). Returns (bytes in, bytes out)."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    with h5py.File(src, "r") as fi, h5py.File(tmp, "w") as fo:
        for k, v in fi.attrs.items():
            fo.attrs[k] = v
        for name in fi:
            if name != "AuxiliaryData":
                fi.copy(name, fo)
                continue
            aux = fo.create_group("AuxiliaryData")
            for sub in fi["AuxiliaryData"]:
                if sub != "Cache":
                    fi.copy(f"AuxiliaryData/{sub}", aux)
                    continue
                cache = aux.create_group("Cache")
                for c in fi["AuxiliaryData/Cache"]:
                    if c not in DROP_CACHE:
                        fi.copy(f"AuxiliaryData/Cache/{c}", cache)
    tmp.replace(dst)
    return src.stat().st_size, dst.stat().st_size
