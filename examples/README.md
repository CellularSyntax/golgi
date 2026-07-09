# golgi examples

Runnable scripts that drive the headless [`golgi.Study`](https://github.com/CellularSyntax/golgi/wiki/Python-API)
API end to end.

| Script | What it does |
|---|---|
| [`recruitment_sweep.py`](recruitment_sweep.py) | Full image-to-recruitment pipeline: load nerve → mesh → bipolar cuff → anisotropic FEM → fiber trajectories → amplitude sweep → recruitment-curve PNG → reproducible study bundle. Runs on a built-in synthetic nerve with no data required. |

```bash
# Synthetic nerve (no data needed):
python examples/recruitment_sweep.py

# Your own geometry:
python examples/recruitment_sweep.py --nerve /path/to/nerve.stl --project /tmp/study
```

The compute stages need the FEniCSx solver stack — see
[Installation](https://github.com/CellularSyntax/golgi/wiki/Installation). The fast, solver-free API
guards live in [`tests/test_headless_api.py`](../tests/test_headless_api.py).
