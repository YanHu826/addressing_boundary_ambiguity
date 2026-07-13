#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-build}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTAINER_CLI="${CONTAINER_CLI:-docker}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-${PROJECT_DIR}/Dockerfile}"

LOCAL_IMAGE="${LOCAL_IMAGE:-addressing_boundary_ambiguity:cu126}"
REGISTRY_HOST="${REGISTRY_HOST:-docker.io}"
REGISTRY_NAMESPACE="${REGISTRY_NAMESPACE:-yanhu826}"
REMOTE_REPOSITORY="${REMOTE_REPOSITORY:-addressing_boundary_ambiguity}"
REMOTE_TAG="${REMOTE_TAG:-${LOCAL_IMAGE##*:}}"
REMOTE_IMAGE="${REMOTE_IMAGE:-${REGISTRY_HOST}/${REGISTRY_NAMESPACE}/${REMOTE_REPOSITORY}:${REMOTE_TAG}}"

print_help() {
  cat <<EOF
Usage:
  bash scripts/build_image.sh build
  bash scripts/build_image.sh push
  bash scripts/build_image.sh build-push
  bash scripts/build_image.sh print

Defaults:
  LOCAL_IMAGE=${LOCAL_IMAGE}
  REMOTE_IMAGE=${REMOTE_IMAGE}

Examples:
  sudo bash scripts/build_image.sh build
  sudo bash scripts/build_image.sh push
  sudo bash scripts/build_image.sh build-push
EOF
}

build_image() {
  "${CONTAINER_CLI}" build -f "${DOCKERFILE_PATH}" -t "${LOCAL_IMAGE}" "${PROJECT_DIR}"
}

tag_image() {
  "${CONTAINER_CLI}" tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"
}

push_image() {
  tag_image
  "${CONTAINER_CLI}" push "${REMOTE_IMAGE}"
}

case "${MODE}" in
  build)
    build_image
    ;;
  tag)
    tag_image
    ;;
  push)
    push_image
    ;;
  build-push)
    build_image
    push_image
    ;;
  print)
    printf '%s\n' "${REMOTE_IMAGE}"
    ;;
  help|-h|--help)
    print_help
    ;;
  *)
    printf 'Unknown mode: %s\n' "${MODE}" >&2
    print_help >&2
    exit 1
    ;;
esac
