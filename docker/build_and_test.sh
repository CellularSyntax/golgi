#!/usr/bin/env bash
# Build the golgi Docker image, run the test suite + the end-to-end example +
# the benchmark inside it, and save the image as a tarball for archiving
# (e.g. on Zenodo).
#
#   docker/build_and_test.sh                 # native platform, tag golgi:<version>
#   docker/build_and_test.sh --platform linux/amd64
#   OUT=/path/to/logs docker/build_and_test.sh
#
# Everything (build log, test log, benchmark JSON, lock files copied out of the
# image, image tarball + sha256) lands in $OUT (default: docker/out/).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
VERSION="$(sed -nE 's/^version *= *"([^"]+)"/\1/p' "$ROOT/pyproject.toml")"
IMAGE="${IMAGE:-golgi:${VERSION}}"
OUT="${OUT:-$HERE/out}"
PLATFORM_ARGS=()
FIBERS="${FIBERS:-12}"
PROFILE="${PROFILE:-full}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM_ARGS=(--platform "$2"); shift 2 ;;
    --fibers) FIBERS="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --no-save) NO_SAVE=1; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
mkdir -p "$OUT"
STATUS="$OUT/status.txt"
: > "$STATUS"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

# The Docker daemon needs to be reachable from this shell.
docker version > "$OUT/docker_version.txt" 2>&1 || { log "docker daemon not reachable"; exit 1; }

log "1/6 building $IMAGE (${PLATFORM_ARGS[*]:-native})"
docker build "${PLATFORM_ARGS[@]}" -t "$IMAGE" "$ROOT" 2>&1 | tee "$OUT/build.log" | tail -n 3
log "build OK"

RUN=(docker run --rm "${PLATFORM_ARGS[@]}" -e GOLGI_PROJECTS_ROOT=/tmp/golgi_projects "$IMAGE")

log "2/6 versions inside the image"
"${RUN[@]}" python - > "$OUT/versions.json" <<'EOF'
import json, platform, importlib.metadata as m
pk = ["golgi","fenics-dolfinx","petsc4py","mpi4py","gmsh","tetgen","vtk","pyvista",
      "neuron","pyfibers","trame","trame-vtk","trame-vuetify","numpy","scipy","meshio","h5py","psutil"]
print(json.dumps({"platform": platform.platform(), "python": platform.python_version(),
                  **{p: m.version(p) for p in pk}}, indent=2))
EOF
cat "$OUT/versions.json"

log "3/6 copying resolved lock files out of the image"
CID=$(docker create "${PLATFORM_ARGS[@]}" "$IMAGE")
docker cp "$CID:/opt/golgi/." "$OUT/image_root_tmp" >/dev/null
docker rm "$CID" >/dev/null
cp "$OUT"/image_root_tmp/environment.*.lock "$OUT"/image_root_tmp/requirements-pip.*.txt "$OUT"/ 2>/dev/null || true
rm -rf "$OUT/image_root_tmp"
ls "$OUT"/environment.*.lock

log "4/6 running pytest inside the image"
set +e
docker run --rm "${PLATFORM_ARGS[@]}" -e GOLGI_PROJECTS_ROOT=/tmp/golgi_projects \
  "$IMAGE" python -m pytest -q -rs --durations=5 /opt/golgi/tests 2>&1 | tee "$OUT/pytest.log"
RC=${PIPESTATUS[0]}
set -e
log "pytest exit code $RC"
[[ $RC -eq 0 ]] || { log "pytest FAILED"; exit 1; }

log "5/6 running the end-to-end example + benchmark inside the image ($FIBERS fibers)"
docker run --rm "${PLATFORM_ARGS[@]}" -e GOLGI_PROJECTS_ROOT=/tmp/golgi_projects \
  -v "$OUT:/out" -w /tmp "$IMAGE" \
  bash -lc "python /opt/golgi/examples/benchmark.py --fibers $FIBERS --profile $PROFILE --project /tmp/bench_project --out /out/benchmark_docker.json --label docker-$(uname -m) \
            && golgi replay /tmp/bench_project_study.zip --json > /out/replay_docker.json \
            && echo REPLAY_OK" 2>&1 | tee "$OUT/benchmark_docker.log"
grep -q REPLAY_OK "$OUT/benchmark_docker.log" || { log "example/replay FAILED"; exit 1; }
log "example + bundle verification OK"

if [[ -z "${NO_SAVE:-}" ]]; then
  log "6/6 saving image tarball"
  TAR="$OUT/golgi-${VERSION}-$(uname -m)-docker.tar"
  docker save "$IMAGE" -o "$TAR"
  gzip -f "$TAR"
  shasum -a 256 "$TAR.gz" > "$TAR.gz.sha256"
  ls -lh "$TAR.gz"; cat "$TAR.gz.sha256"
fi
log "ALL DONE — results in $OUT"
