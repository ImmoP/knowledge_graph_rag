# Local Infrastructure Specification

Resolution of [#3 — Decide the local infrastructure shape](https://github.com/ImmoP/knowledge_graph_rag/issues/3). This document is the **contract** the running stack must satisfy. It is not the implementing ticket — that goes to `/to-tickets` once the map is cleared.

The local Docker topology is the project's only infrastructure. There is no cloud-managed Neo4j, no cloud-managed Postgres, no queues, no orchestrators, no caches. AI/model providers (per [#16](https://github.com/ImmoP/knowledge_graph_rag/issues/16)) are an explicit exception to that restriction — adapters may target either local model runtimes or remote model APIs — but they are not infrastructure in the sense of this ticket; they are provider configuration.

## 1. Service list

Exactly two services. No more.

| Service       | Role                                                                                                    | Owns                                                                                                                                          |
|---------------|---------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `neo4j`       | Graph store for `Document`, `Chunk`, entity, and relation nodes/edges per [#9](https://github.com/ImmoP/knowledge_graph_rag/issues/9). | The knowledge graph itself: nodes for documents and chunks, nodes for ontology-defined entities, edges for ontology-defined relations, plus the join edges `(:Document)-[:HAS_CHUNK]->(:Chunk)` and `(:Chunk)-[:MENTIONS]->(:Entity)`. |
| `pgvector`    | Vector store for chunk embeddings per [#10](https://github.com/ImmoP/knowledge_graph_rag/issues/10).    | The `chunks` table: `chunk_id` (PK), `doc_id`, `section_path` (JSONB), `chunk_index`, `text`, `token_estimate`, `embedding` (`vector(N)` where N = embedding dimension pinned in #10). |

**What neither service owns:**
- The chunk manifest on disk (`data/derived/chunks.jsonl`). The manifest is the source of truth; both stores are derived from it. Stores can be wiped and re-derived without losing the corpus.
- Extraction results (`data/extractions.jsonl` per [#7](https://github.com/ImmoP/knowledge_graph_rag/issues/7)). Same property: derived stores; wipeable.
- AI/model state (provider caches, downloaded model weights, prompt templates). Provider concerns are not infrastructure concerns.

## 2. Pinned image versions

Versions are pinned by tag, not by digest. Resolved on 2026-09-10. The implementing ticket writes the `docker-compose.yml` at the repo root with these exact tags.

| Service    | Image                                                                  | Version as of 2026-09-10 |
|------------|------------------------------------------------------------------------|--------------------------|
| `neo4j`    | `neo4j:5.26-community-ubi10`                                           | current community stable |
| `pgvector` | `pgvector/pgvector:0.8.6-pg17-trixie`                                  | pgvector 0.8.6 on PostgreSQL 17 (trixie base) |

**Rationale:**
- **Neo4j 5.26 community** is the most recent community-edition release as of the decision date. UBI10 (Universal Base Image 10) is Red Hat's base; it is the image Neo4j publishes for current versions. Community edition is sufficient (no clustering, no multi-db beyond the default).
- **pgvector 0.8.6 on PostgreSQL 17** is the most recent pgvector release supporting Postgres 17. Postgres 17 is the current major; pgvector 0.8.x supports pg14–pg18. The trixie base (Debian 13) is the most recent published base; bookworm (Debian 12) is also available if the user prefers LTS base.

**Forward-compatibility note:** the policy is "pin current stable; bump in a follow-up commit when the current version goes EOL or has a security advisory." Both versions are expected to be supported for ≥18 months. The implementing ticket documents the pin in `docker-compose.yml` with a comment naming the date and source so a future bump is one-line.

## 3. Network shape

A single private bridge network named `rag-net`. Both services attach to it. The application code (running on the host) connects to the published ports; the two services can address each other by service name (`neo4j`, `pgvector`).

The bridge is **not** exposed to the host beyond the explicitly published ports below. There is no host networking, no port ranges, no extra interfaces.

### 3.1 Published ports

| Service    | Container port | Host port  | Use                                  |
|------------|----------------|------------|--------------------------------------|
| `neo4j`    | 7474 (HTTP)    | 7474       | Neo4j Browser (development only)     |
| `neo4j`    | 7687 (Bolt)    | 7687       | Bolt protocol — application queries  |
| `pgvector` | 5432           | 5433       | PostgreSQL — application queries     |

Port 5433 on the host (not 5432) avoids clashes with any system-local Postgres. Inside the bridge, services use the canonical 5432.

## 4. Persistent named volumes

Three named volumes. All survive `docker compose down` and `docker compose up`. The implementing ticket MUST NOT use `docker compose down -v` in any test or CI script — that would delete data.

| Volume           | Mounted at                       | Contents                                                                                    |
|------------------|----------------------------------|---------------------------------------------------------------------------------------------|
| `neo4j_data`     | `/data`                          | Neo4j graph store on disk (the actual database files).                                      |
| `neo4j_logs`     | `/logs`                          | Neo4j logs. Kept on a separate volume so log churn cannot fill the data volume.             |
| `pgvector_data`  | `/var/lib/postgresql/data`       | PostgreSQL data directory (pgvector's actual data files including the `chunks` table).      |

Named volumes (not bind mounts) are the right choice here because:
- They survive `down` and `up` without host-path coupling.
- Docker manages permissions; bind mounts on a CI host or new laptop cause permission issues.
- The user does not need to inspect or back up the data directory directly (the manifest + extraction files on disk are the human-readable artifacts).

## 5. Credentials and env-var policy

All credentials are read from `.env` at the repo root. None appear in `docker-compose.yml`. The compose file references env vars via `${VAR_NAME}` and the user's `.env` provides the values.

### 5.1 Env-var names pinned

| Env var              | Required | Default in compose                              | Used by                                |
|----------------------|----------|-------------------------------------------------|----------------------------------------|
| `NEO4J_PASSWORD`     | yes      | (none — must be set in `.env`)                  | `neo4j` initial password and Bolt auth  |
| `NEO4J_USER`         | no       | `neo4j`                                         | Neo4j auth username                    |
| `NEO4J_URI`          | no       | `bolt://localhost:7687`                         | App-side Neo4j connection URI          |
| `PGVECTOR_USER`      | yes      | (none — must be set in `.env`)                  | Postgres user                          |
| `PGVECTOR_PASSWORD`  | yes      | (none — must be set in `.env`)                  | Postgres password                      |
| `PGVECTOR_DB`        | no       | `rag`                                           | Database name (created at first boot)  |
| `PGVECTOR_HOST`      | no       | `localhost`                                     | App-side Postgres host                 |
| `PGVECTOR_PORT`      | no       | `5433`                                          | App-side Postgres port                 |

`NEO4J_PASSWORD` already exists in `.env.example` (currently empty). The implementing ticket adds `PGVECTOR_USER`, `PGVECTOR_PASSWORD`, `PGVECTOR_DB` to `.env.example` under a clearly-commented `# ---- Local infrastructure ----` section, distinct from the AI-provider block documented in the provider-abstractions spec.

### 5.2 What never appears in compose

- No password or API key as a literal string.
- No `environment: PASSWORD: mysecret` lines.
- No `secrets:` block referencing in-repo secret files.
- No external secret manager. Credentials live in `.env`, which is gitignored.

If a future contributor wants to add a credential, the rule is: read it from `.env`, document the env var name in this spec, and reference it via `${VAR}` in compose. Anything else is a violation of the policy.

## 6. Health-check policy

Each service has a `healthcheck` block. The stack is considered ready when both `neo4j` and `pgvector` report `healthy`.

### 6.1 Neo4j

```yaml
healthcheck:
  test: ["CMD-SHELL", "wget -q --tries=1 --spider http://localhost:7474 || exit 1"]
  interval: 10s
  timeout: 5s
  retries: 10
  start_period: 60s
```

The HTTP endpoint at port 7474 returns 200 when Neo4j is accepting Bolt queries. `start_period: 60s` accounts for the first-boot delay during which Neo4j initializes the on-disk store.

### 6.2 pgvector

```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
  interval: 5s
  timeout: 5s
  retries: 10
  start_period: 30s
```

`pg_isready` is shipped inside the official postgres image and is the canonical readiness probe.

### 6.3 What health checks do NOT do

- They do not test the Bolt protocol end-to-end (Neo4j HTTP up ≠ Bolt accepting queries in every race).
- They do not test database connectivity from the host.
- They do not validate that the schema (nodes, constraints, vector extension) is installed — that's the implementing ticket's responsibility, exercised by the resolution criterion in §8.

## 7. Service configuration

### 7.1 Neo4j

- Single instance (no clustering, no secondaries). The corpus is small (182 documents, a few thousand chunks, a few thousand entities); a single instance handles it.
- Plugins: none beyond defaults. The `apoc` plugin is **not** loaded — none of the planned Cypher queries need it.
- Memory: leave the default JVM heap sizing. The implementing ticket does not pin `NEO4J_HEAP_MEMORY`; Neo4j's auto-sizing handles the corpus comfortably.
- Auth: enabled. Bolt requires `neo4j` + `NEO4J_PASSWORD`. The HTTP endpoint requires auth as well (no anonymous browser access in the first run).
- `NEO4J_PLUGINS`: unset (no plugins).

### 7.2 pgvector

- Single instance. Postgres 17 with pgvector 0.8.6.
- `POSTGRES_DB=rag` (driven by `PGVECTOR_DB`), `POSTGRES_USER=postgres` (driven by `PGVECTOR_USER`).
- No replication, no backups. The corpus is reconstructible from the on-disk manifest and the raw LaTeX; a backup is just "re-derive."
- The pgvector extension is created by the implementing ticket on first connect (`CREATE EXTENSION IF NOT EXISTS vector;`). The implementing ticket's bootstrap script creates the database, the extension, and the `chunks` table schema.

## 8. Resolution criterion for the implementing ticket

The implementing ticket closes only when all of the following succeed, in order:

1. **`docker compose up -d` brings both services to a healthy state.** `docker compose ps` shows both `neo4j` and `pgvector` with status `(healthy)` within 2 minutes.
2. **Host-side reachability via `.env` credentials:**
   - Neo4j Bolt: a Python script using the `neo4j` driver connects with `${NEO4J_USER}` + `${NEO4J_PASSWORD}` from `.env`, runs `RETURN 1`, gets `1`.
   - pgvector: a Python script using `psycopg` connects with `${PGVECTOR_USER}` + `${PGVECTOR_PASSWORD}` from `.env`, runs `SELECT 1`, gets `1`.
   - The Python scripts read credentials from `.env` via `python-dotenv` (already a project dep).
3. **`docker compose down && docker compose up -d` twice** — the stack comes back healthy both times. No data written between cycles is lost. The implementing ticket writes a sentinel value (e.g. a `(:Sentinel {ts: datetime()})` node in Neo4j, a `pgvector_smoke` row in Postgres) before each cycle and asserts it persists across the cycle.
4. **No secrets in compose:** `grep -iE "password|api_key|secret" docker-compose.yml` returns only `${...}`-style references, never a literal credential.

## 9. New dependencies the implementing ticket adds

The implementing ticket extends `requirements.txt` to add the Python drivers needed by the smoke-test scripts and by later retrieval/ingestion tickets:

- `neo4j` — official Neo4j Python driver (Bolt protocol).
- `psycopg[binary]` — Postgres driver, with the binary build so no system-level `libpq` is required.

Both are the canonical first-party drivers. No ORM (no SQLAlchemy, no neo4j-graphql). Application code talks to the drivers directly.

The implementing ticket also adds `neo4j` and `psycopg` to any development extras list if the project uses one (it currently does not).

## 10. What this spec leaves open (implementing ticket owns)

These are explicitly not part of the architecture contract:

- **Exact `docker-compose.yml` formatting** (YAML layout, key ordering). The spec gives the substantive fields; the implementing ticket writes idiomatic YAML.
- **Postgres extensions beyond `vector`.** No `pg_trgm`, no `uuid-ossp` unless a later ticket's spec demands them.
- **Pre-loading the `chunks` schema.** The bootstrap script that creates the extension and the `chunks` table is the implementing ticket's job; it is referenced by the [#10](https://github.com/ImmoP/knowledge_graph_rag/issues/10) implementation but its shape is fixed here.
- **CI / pre-commit hooks** that run `docker compose up` automatically. Out of scope for the implementing ticket.
- **Backup/restore scripts.** The project does not need them; the corpus is re-derivable.
- **TLS / SSL on Bolt or Postgres.** Local-only stack; loopback connections over the bridge are not encrypted. Production hardening is out of scope.

## 11. What this spec forbids

These are explicit architectural commitments; adding them requires re-opening the map, not just editing the spec:

- **A third service.** No Redis, no queues, no orchestration, no observability stack, no admin UI beyond Neo4j Browser.
- **A hosted database service** of any kind (no Neo4j Aura, no Supabase, no Neon, no RDS).
- **Bind mounts for data directories.** Named volumes only.
- **Default credentials.** `neo4j/neo4j` or `postgres/postgres` are not acceptable initial passwords; the env vars in `.env` must contain real values.
- **Cross-service auth sharing.** Neo4j and Postgres have separate credential sets; they are not unified.

---

This spec resolves ticket #3. It is referenced by #9 (Neo4j ingestion connects via `NEO4J_URI`), #10 (vector retrieval connects via `PGVECTOR_*`), #11 (graph retrieval connects via `NEO4J_URI`), and the implementing ticket for the Docker stack itself.
