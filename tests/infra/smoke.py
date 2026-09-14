"""Live verification for the issue #18 local stores.

Run ``python -m tests.infra.smoke`` after the Compose services are healthy.
Configuration defaults to the repository's gitignored ``.env``. The default
check performs one real query over each published host port:

* Neo4j Bolt: ``RETURN 1``
* PostgreSQL: ``SELECT 1``

The sentinel actions are used by ``scripts/infra_up.sh`` to prove both named
data volumes survive individual container restarts. They create only
short-lived smoke state, never application schema.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
REQUIRED_ENV_VARS = (
    "NEO4J_PASSWORD",
    "PGVECTOR_USER",
    "PGVECTOR_PASSWORD",
)
SENTINEL_TABLE = "_issue18_infra_smoke"


def _abort(reason: str) -> NoReturn:
    print(f"INFRA SMOKE: FAIL ({reason})", file=sys.stderr)
    raise SystemExit(1)


def _load_config(env_file: Path) -> dict[str, str]:
    if not env_file.is_file():
        _abort(f"missing {env_file}")

    # Compose reads the same env file. Override inherited values so the
    # host-side probes cannot accidentally use different credentials.
    load_dotenv(env_file, override=True)

    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        _abort(f"blank or missing values in env file: {', '.join(missing)}")

    return {
        "neo4j_uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "neo4j_user": os.environ.get("NEO4J_USER", "neo4j"),
        "neo4j_password": os.environ["NEO4J_PASSWORD"],
        "pgvector_host": os.environ.get("PGVECTOR_HOST", "localhost"),
        "pgvector_port": os.environ.get("PGVECTOR_PORT", "5433"),
        "pgvector_user": os.environ["PGVECTOR_USER"],
        "pgvector_password": os.environ["PGVECTOR_PASSWORD"],
        "pgvector_db": os.environ.get("PGVECTOR_DB", "rag"),
    }


def _verify_neo4j(config: dict[str, str], action: str, token: str | None) -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        _abort("neo4j driver is not installed; install requirements.txt")

    try:
        with GraphDatabase.driver(
            config["neo4j_uri"],
            auth=(config["neo4j_user"], config["neo4j_password"]),
            connection_timeout=10,
        ) as driver:
            driver.verify_connectivity()
            with driver.session(database="neo4j") as session:
                if action == "check":
                    result = session.run("RETURN 1 AS value").single(strict=True)["value"]
                    if result != 1:
                        _abort(f"Neo4j RETURN 1 produced {result!r}")
                elif action == "write-sentinel":
                    session.run(
                        "MERGE (s:_Issue18InfraSmoke {token: $token}) RETURN s.token",
                        token=token,
                    ).consume()
                elif action == "assert-sentinel":
                    count = session.run(
                        "MATCH (s:_Issue18InfraSmoke {token: $token}) "
                        "RETURN count(s) AS count",
                        token=token,
                    ).single(strict=True)["count"]
                    if count != 1:
                        _abort(f"Neo4j persistence sentinel count was {count!r}, expected 1")
                else:
                    session.run(
                        "MATCH (s:_Issue18InfraSmoke {token: $token}) DELETE s",
                        token=token,
                    ).consume()
    except SystemExit:
        raise
    except Exception as exc:
        _abort(f"Neo4j operation failed: {type(exc).__name__}: {exc}")

    print(f"INFRA SMOKE: Neo4j {action} OK")


def _verify_pgvector(config: dict[str, str], action: str, token: str | None) -> None:
    try:
        import psycopg
    except ImportError:
        _abort("psycopg driver is not installed; install requirements.txt")

    try:
        with psycopg.connect(
            host=config["pgvector_host"],
            port=config["pgvector_port"],
            user=config["pgvector_user"],
            password=config["pgvector_password"],
            dbname=config["pgvector_db"],
            connect_timeout=10,
        ) as connection:
            with connection.cursor() as cursor:
                if action == "check":
                    cursor.execute("SELECT 1")
                    row = cursor.fetchone()
                    if row != (1,):
                        _abort(f"PostgreSQL SELECT 1 produced {row!r}")
                elif action == "write-sentinel":
                    cursor.execute(
                        f"CREATE TABLE IF NOT EXISTS {SENTINEL_TABLE} "
                        "(token text PRIMARY KEY)"
                    )
                    cursor.execute(
                        f"INSERT INTO {SENTINEL_TABLE} (token) VALUES (%s) "
                        "ON CONFLICT (token) DO NOTHING",
                        (token,),
                    )
                elif action == "assert-sentinel":
                    cursor.execute(
                        f"SELECT count(*) FROM {SENTINEL_TABLE} WHERE token = %s",
                        (token,),
                    )
                    row = cursor.fetchone()
                    if row != (1,):
                        _abort(f"PostgreSQL persistence sentinel count was {row!r}, expected (1,)")
                else:
                    cursor.execute(
                        f"DELETE FROM {SENTINEL_TABLE} WHERE token = %s",
                        (token,),
                    )
                    cursor.execute(f"SELECT count(*) FROM {SENTINEL_TABLE}")
                    if cursor.fetchone() == (0,):
                        cursor.execute(f"DROP TABLE {SENTINEL_TABLE}")
    except SystemExit:
        raise
    except Exception as exc:
        _abort(f"PostgreSQL operation failed: {type(exc).__name__}: {exc}")

    print(f"INFRA SMOKE: PostgreSQL {action} OK")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=("check", "write-sentinel", "assert-sentinel", "delete-sentinel"),
        default="check",
    )
    parser.add_argument("token", nargs="?")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="credential file (default: repository .env)",
    )
    args = parser.parse_args()
    if args.action != "check" and not args.token:
        parser.error(f"{args.action} requires a token")
    if args.action == "check" and args.token:
        parser.error("check does not accept a token")
    return args


def main() -> int:
    args = _parse_args()
    config = _load_config(args.env_file)
    _verify_neo4j(config, args.action, args.token)
    _verify_pgvector(config, args.action, args.token)
    print("INFRA SMOKE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
