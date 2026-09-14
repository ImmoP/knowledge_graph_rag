#!/usr/bin/env bash
# Start issue #18's two-service local stack, wait for Docker healthchecks,
# verify both host protocols, then prove state survives container restarts.

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly TIMEOUT_SECONDS="${INFRA_HEALTH_TIMEOUT_SECONDS:-120}"
readonly POLL_SECONDS="${INFRA_HEALTH_POLL_SECONDS:-2}"
readonly PYTHON_BIN="${PYTHON_BIN:-python3}"
readonly ENV_FILE="${INFRA_ENV_FILE:-.env}"

cd "${REPO_ROOT}"

if ! command -v docker >/dev/null 2>&1; then
  printf 'INFRA: FAIL (docker is not installed or not on PATH)\n' >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  printf 'INFRA: FAIL (Docker Compose v2 is unavailable)\n' >&2
  exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  printf 'INFRA: FAIL (missing %s/%s; copy .env.example and set credentials)\n' \
    "${REPO_ROOT}" "${ENV_FILE}" >&2
  exit 1
fi
export COMPOSE_ENV_FILES="${ENV_FILE}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  printf 'INFRA: FAIL (%s is not installed or not on PATH)\n' "${PYTHON_BIN}" >&2
  exit 1
fi

container_health() {
  local service="$1"
  local container_id
  container_id="$(docker compose ps -q "${service}")"
  if [[ -z "${container_id}" ]]; then
    printf 'missing'
    return
  fi
  docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}}' "${container_id}"
}

wait_for_health() {
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  local neo4j_health='unknown'
  local pgvector_health='unknown'

  while (( SECONDS < deadline )); do
    neo4j_health="$(container_health neo4j)"
    pgvector_health="$(container_health pgvector)"

    if [[ "${neo4j_health}" == "healthy" && "${pgvector_health}" == "healthy" ]]; then
      printf 'INFRA: healthchecks OK (neo4j=%s, pgvector=%s)\n' \
        "${neo4j_health}" "${pgvector_health}"
      return
    fi

    if [[ "${neo4j_health}" == "unhealthy" || "${pgvector_health}" == "unhealthy" ]]; then
      printf 'INFRA: FAIL (unhealthy service: neo4j=%s, pgvector=%s)\n' \
        "${neo4j_health}" "${pgvector_health}" >&2
      docker compose ps >&2
      exit 1
    fi

    sleep "${POLL_SECONDS}"
  done

  printf 'INFRA: FAIL (healthcheck timeout after %ss: neo4j=%s, pgvector=%s)\n' \
    "${TIMEOUT_SECONDS}" "${neo4j_health}" "${pgvector_health}" >&2
  docker compose ps >&2
  exit 1
}

printf 'INFRA: starting neo4j and pgvector\n'
docker compose up -d
wait_for_health
"${PYTHON_BIN}" -m tests.infra.smoke --env-file "${ENV_FILE}"

readonly sentinel='issue18-persistence-check'
cleanup_sentinel() {
  "${PYTHON_BIN}" -m tests.infra.smoke delete-sentinel "${sentinel}" --env-file "${ENV_FILE}" >/dev/null 2>&1 || true
}
trap cleanup_sentinel EXIT
# Remove any marker left by an interrupted prior run before writing this one.
cleanup_sentinel

"${PYTHON_BIN}" -m tests.infra.smoke write-sentinel "${sentinel}" --env-file "${ENV_FILE}"
for cycle in 1 2; do
  printf 'INFRA: recreating containers (cycle %s/2) to verify named-volume persistence\n' "${cycle}"
  docker compose down
  docker compose up -d
  wait_for_health
  "${PYTHON_BIN}" -m tests.infra.smoke assert-sentinel "${sentinel}" --env-file "${ENV_FILE}"
done
"${PYTHON_BIN}" -m tests.infra.smoke delete-sentinel "${sentinel}" --env-file "${ENV_FILE}"
trap - EXIT

printf 'INFRA: PASS\n'
