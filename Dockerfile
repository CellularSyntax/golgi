# syntax=docker/dockerfile:1
# golgi — reproducible container image (GUI + Python API + CLI).
#
#   docker build -t golgi:1.1.0 .
#   docker run --rm -p 8080:8080 -v "$PWD/golgi_data:/data" golgi:1.1.0
#   → open http://localhost:8080
#
# Headless use (API / CLI / tests):
#   docker run --rm -v "$PWD:/work" -w /work golgi:1.1.0 python examples/recruitment_sweep.py
#   docker run --rm golgi:1.1.0 golgi replay /data/my_study.zip
#   docker run --rm golgi:1.1.0 pytest -q
#
# The image pins the same scientific stack as environment.yml (conda-forge:
# FEniCSx/DOLFINx 0.10, PETSc, Gmsh 4.15.2, VTK 9.6.1, NEURON 9.0.1; PyPI:
# PyFibers 0.8.5, Trame 3.12, TetGen 0.8.4) and records the fully resolved
# package list in /opt/golgi/environment.<platform>.lock for provenance.
# The optional AxonML GPU backend is NOT included (separate Duke licence).

ARG MICROMAMBA_VERSION=2.3.0
FROM mambaorg/micromamba:${MICROMAMBA_VERSION}-ubuntu24.04 AS base

USER root
# Runtime libraries for VTK/Gmsh off-screen rendering and a display shim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglu1-mesa libegl1 libosmesa6 libxrender1 libxcursor1 \
        libxft2 libxinerama1 libxi6 libxext6 libsm6 libice6 xvfb \
        ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/golgi /data && chown -R $MAMBA_USER:$MAMBA_USER /opt/golgi /data
USER $MAMBA_USER
WORKDIR /opt/golgi

# --- 1. compiled scientific core from conda-forge (cached layer) -----------
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
# Use only the conda part of environment.yml here; the pip layer follows once
# the sources are in the image (keeps this expensive layer cacheable).
RUN sed '/- pip:/,$d' /tmp/environment.yml > /tmp/conda-only.yml && \
    micromamba install -y -n base -f /tmp/conda-only.yml && \
    micromamba clean --all --yes

# --- 2. golgi + pure-Python dependencies from PyPI --------------------------
ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/golgi
RUN python -m pip install --no-cache-dir \
        "pyfibers==0.8.5" "trame==3.12.0" "trame-client==3.12.1" \
        "trame-server==3.12.0" "trame-vtk==2.11.8" "trame-vuetify==3.2.2" \
        "trame-plotly==3.1.2" "trame-common==1.2.3" "wslink==2.5.6" \
        "tetgen==0.8.4" psutil && \
    python -m pip install --no-cache-dir -e /opt/golgi

# --- 3. compile PyFibers' NEURON mechanisms ---------------------------------
RUN set -e; \
    export CXX="$(ls ${CONDA_PREFIX}/bin/*-conda-linux-gnu-g++ | head -1)"; \
    export CC="$(ls ${CONDA_PREFIX}/bin/*-conda-linux-gnu-gcc | head -1)"; \
    MOD="$(python -c 'import importlib.util as u;print(u.find_spec("pyfibers").submodule_search_locations[0])')/MOD"; \
    test -d "$MOD" || { echo "PyFibers MOD dir not found: $MOD"; exit 1; }; \
    cd "$MOD" && nrnivmodl > /tmp/nrnivmodl.log 2>&1 || { tail -40 /tmp/nrnivmodl.log; exit 1; }; \
    python -c "import pyfibers; print('pyfibers', pyfibers.__version__, 'mechanisms OK')"

# --- 4. record the resolved environment for provenance ----------------------
RUN micromamba env export -n base --explicit --md5 > /opt/golgi/environment.$(uname -m).lock 2>/dev/null || \
    micromamba list -n base --explicit --md5 > /opt/golgi/environment.$(uname -m).lock; \
    python -m pip freeze --exclude-editable > /opt/golgi/requirements-pip.$(uname -m).txt

# --- runtime configuration ---------------------------------------------------
ENV GOLGI_PROJECTS_ROOT=/data \
    GOLGI_HOST=0.0.0.0 \
    GOLGI_NO_BROWSER=1 \
    GOLGI_IN_CONTAINER=1 \
    PYVISTA_OFF_SCREEN=true \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]
EXPOSE 8080

LABEL org.opencontainers.image.title="golgi" \
      org.opencontainers.image.description="Open-source image-to-recruitment modeling of peripheral nerve stimulation (GUI, Python API, CLI)" \
      org.opencontainers.image.source="https://github.com/CellularSyntax/golgi" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later"

# Default: launch the GUI server. Any other command (python, pytest, golgi
# export|import|replay, bash) can be passed to `docker run`.
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["golgi", "--host", "0.0.0.0", "--port", "8080", "--no-browser"]
