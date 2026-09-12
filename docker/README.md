# golgi in a container

The [`Dockerfile`](../Dockerfile) at the repository root builds a self-contained image with the
complete scientific stack (FEniCSx/DOLFINx 0.10, PETSc/MUMPS, Gmsh 4.15.2, TetGen 0.8.4, VTK 9.6.1,
NEURON 9.0.1 with PyFibers 0.8.5 mechanisms pre-compiled, Trame 3.12) on Ubuntu 24.04 via
micromamba. The optional AxonML GPU backend is not included (separate Duke University licence).

## Get the image

```bash
# from GitHub Container Registry (built by CI on every tagged release)
docker pull ghcr.io/cellularsyntax/golgi:v1.1.0

# or from the Zenodo archive
docker load -i golgi-1.1.0-x86_64-docker.tar.gz

# or build it yourself (≈10–20 min, mostly conda downloads)
docker build -t golgi:1.1.0 .
```

## Run

| Task | Command |
|---|---|
| GUI | `docker run --rm -p 8080:8080 -v "$PWD/golgi_data:/data" golgi:1.1.0` → http://localhost:8080 |
| GUI via compose | `docker compose -f docker/docker-compose.yml up` |
| Python API script | `docker run --rm -v "$PWD:/work" -w /work golgi:1.1.0 python my_study.py` |
| Example study | `docker run --rm -v "$PWD:/work" -w /work golgi:1.1.0 python /opt/golgi/examples/recruitment_sweep.py` |
| Verify a bundle | `docker run --rm -v "$PWD:/work" -w /work golgi:1.1.0 golgi replay study.golgi.zip` |
| Test suite | `docker run --rm golgi:1.1.0 pytest -q /opt/golgi/tests` |
| Benchmark | `docker run --rm -v "$PWD:/work" -w /work golgi:1.1.0 python /opt/golgi/examples/benchmark.py` |
| Shell | `docker run --rm -it golgi:1.1.0 bash` |

Projects live in `/data` inside the container (`GOLGI_PROJECTS_ROOT`); mount a host directory there
to keep them. The image runs as the unprivileged `mambauser`; if a mounted directory is not writable,
add `--user $(id -u):$(id -g)`.

The FEM solve and the NEURON fiber simulations run in worker subprocesses and use all CPUs Docker
grants the container; on Docker Desktop raise the CPU/memory limits (Settings → Resources) for large
meshes. `OMP_NUM_THREADS=1` is set by default because the workers parallelise across processes.

## Build, test and archive in one step

```bash
docker/build_and_test.sh                       # native platform
docker/build_and_test.sh --platform linux/amd64
```

builds the image, runs the full test suite, the end-to-end example plus the benchmark inside it,
verifies the exported bundle with `golgi replay`, copies the resolved environment lock files out of the
image and writes a `docker save` tarball with its SHA-256 to `docker/out/` — the artefacts we archive
on Zenodo with every release.

## Provenance

`/opt/golgi/environment.<arch>.lock` and `/opt/golgi/requirements-pip.<arch>.txt` inside the image list
every installed package with version and hash.
