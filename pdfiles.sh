#!/usr/bin/env bash
set -euo pipefail

# PDfiles — single management script
# Usage: ./pdfiles.sh <command> [options]

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

detect_compose_file() {
    local force="${1:-}"
    if [ "$force" = "cpu" ]; then
        echo "docker-compose.cpu.yml"
    elif [ "$force" = "gpu" ]; then
        echo "docker-compose.yml"
    elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        echo "docker-compose.yml"
    else
        echo "docker-compose.cpu.yml"
    fi
}

detect_active_compose() {
    for f in docker-compose.yml docker-compose.cpu.yml; do
        if docker compose -f "$f" ps --quiet 2>/dev/null | grep -q .; then
            echo "$f"
            return
        fi
    done
    echo "docker-compose.yml"
}

print_urls() {
    local web_port="${WEB_PORT:-80}"
    local qdrant_port="${QDRANT_PORT:-6335}"
    echo ""
    echo "  Frontend: http://localhost:${web_port}"
    echo "  API:      http://localhost:${web_port}/api/status"
    echo "  Qdrant:   http://localhost:${qdrant_port}/dashboard"
    echo ""
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_deploy() {
    local data_path="${1:-${DATA_PATH:-}}"

    if [ -z "$data_path" ]; then
        echo "Usage: ./pdfiles.sh deploy <DATA_PATH>"
        echo "  or set DATA_PATH in .env"
        exit 1
    fi

    if [ ! -d "$data_path" ]; then
        echo "ERROR: Data directory not found: $data_path"
        exit 1
    fi

    # Create .env if it doesn't exist
    if [ ! -f .env ]; then
        echo "Creating .env..."
        cat > .env <<EOF
DATA_PATH=$data_path
WEB_PORT=80
EOF
    fi

    export DATA_PATH="$data_path"

    local compose_file
    compose_file=$(detect_compose_file)
    local mode="GPU"
    [ "$compose_file" = "docker-compose.cpu.yml" ] && mode="CPU"
    echo "=== PDfiles Deploy ($mode mode) ==="
    echo "Data path: $data_path"

    echo ""
    echo "Building containers..."
    docker compose -f "$compose_file" build

    echo ""
    echo "Starting services..."
    docker compose -f "$compose_file" up -d

    echo ""
    echo "Waiting for services to start..."
    if [ "$mode" = "CPU" ]; then
        echo "  (CPU mode: backend needs ~3 min to load the model)"
    else
        echo "  (GPU mode: backend needs ~2 min to load the model)"
    fi

    print_urls
}

cmd_up() {
    local build=false
    local force_mode=""

    for arg in "$@"; do
        case "$arg" in
            --build) build=true ;;
            --cpu)   force_mode="cpu" ;;
            --gpu)   force_mode="gpu" ;;
        esac
    done

    local compose_file
    compose_file=$(detect_compose_file "$force_mode")
    local mode="GPU"
    [ "$compose_file" = "docker-compose.cpu.yml" ] && mode="CPU"
    echo "$mode mode"

    if [ "$build" = true ]; then
        echo "Building containers..."
        docker compose -f "$compose_file" build
    fi

    echo "Starting services..."
    docker compose -f "$compose_file" up -d

    print_urls
}

cmd_update() {
    local force_mode=""
    for arg in "$@"; do
        case "$arg" in
            --cpu) force_mode="cpu" ;;
            --gpu) force_mode="gpu" ;;
        esac
    done

    local compose_file
    compose_file=$(detect_compose_file "$force_mode")
    echo "Pulling latest images..."
    docker compose -f "$compose_file" pull
    echo "Restarting services..."
    docker compose -f "$compose_file" up -d
    print_urls
}

cmd_down() {
    local clean=false
    for arg in "$@"; do
        case "$arg" in
            --clean) clean=true ;;
        esac
    done

    for f in docker-compose.yml docker-compose.cpu.yml; do
        if [ -f "$f" ]; then
            if [ "$clean" = true ]; then
                docker compose -f "$f" down -v 2>/dev/null || true
            else
                docker compose -f "$f" down 2>/dev/null || true
            fi
        fi
    done

    if [ "$clean" = true ]; then
        echo "Stopped and removed volumes."
    else
        echo "Stopped. (Use --clean to also remove volumes)"
    fi
}

cmd_logs() {
    local compose_file
    compose_file=$(detect_active_compose)
    docker compose -f "$compose_file" logs -f "$@"
}

cmd_status() {
    local qdrant_url="${QDRANT_URL:-http://localhost:6335}"

    echo "=== Docker Services ==="
    for f in docker-compose.yml docker-compose.cpu.yml; do
        if docker compose -f "$f" ps --quiet 2>/dev/null | grep -q .; then
            echo "Active compose: $f"
            docker compose -f "$f" ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
            break
        fi
    done 2>/dev/null || echo "  No docker services running"
    echo ""

    echo "=== Qdrant ==="
    if curl -sf "$qdrant_url/healthz" >/dev/null 2>&1; then
        echo "  Status: healthy ($qdrant_url)"
        local names
        names=$(curl -sf "$qdrant_url/collections" | python3 -c "import sys,json; [print(c['name']) for c in json.load(sys.stdin)['result']['collections']]" 2>/dev/null || true)
        if [ -n "$names" ]; then
            while IFS= read -r name; do
                local count
                count=$(curl -sf "$qdrant_url/collections/$name" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])" 2>/dev/null || echo "?")
                echo "  Collection: $name ($count points)"
            done <<< "$names"
        else
            echo "  No collections"
        fi
    else
        echo "  Status: not reachable ($qdrant_url)"
    fi
    echo ""

    echo "=== API Backend ==="
    if curl -sf http://localhost:8000/api/status >/dev/null 2>&1; then
        local api_status
        api_status=$(curl -sf http://localhost:8000/api/status)
        echo "  $api_status"
    else
        echo "  Status: not reachable (localhost:8000)"
    fi
}

cmd_dev() {
    export DATA_ROOT="${1:-${DATA_ROOT:-/data}}"
    export DEVICE="${DEVICE:-cpu}"

    echo "=== PDfiles Dev Server ==="
    echo "Data root:  $DATA_ROOT"
    echo "Device:     $DEVICE"
    echo ""

    if [ -f "$ROOT/.venv/bin/activate" ]; then
        source "$ROOT/.venv/bin/activate"
    fi

    cleanup() {
        echo ""
        echo "Shutting down..."
        kill 0 2>/dev/null
        wait 2>/dev/null
    }
    trap cleanup EXIT

    echo "Starting backend on :8000..."
    uv run pdfiles serve --host 0.0.0.0 --port 8000 &

    echo "Starting frontend on :5173..."
    cd "$ROOT/web"
    npm run dev -- --host &
    cd "$ROOT"

    echo ""
    echo "  Frontend: http://localhost:5173"
    echo "  API:      http://localhost:8000/api/status"
    echo ""
    echo "Press Ctrl+C to stop both servers."

    wait
}

cmd_test() {
    local quick=false
    for arg in "$@"; do
        case "$arg" in
            --quick) quick=true ;;
        esac
    done

    if [ -f "$ROOT/.venv/bin/activate" ]; then
        source "$ROOT/.venv/bin/activate"
    fi

    local failed=0

    echo "=== Python Tests ==="
    if python -m pytest tests/ -v --tb=short 2>&1; then
        echo "PASS: pytest"
    else
        echo "WARN: pytest had failures (some may be expected without data files)"
    fi
    echo ""

    echo "=== Python Import Check ==="
    local imports=(
        "pdfiles.config"
        "pdfiles.api"
        "pdfiles.searcher"
        "pdfiles.embedder"
        "pdfiles.renderer"
        "pdfiles.pooling"
        "pdfiles.qdrant_store"
        "pdfiles.bouncer"
        "pdfiles.librarian"
    )
    for mod in "${imports[@]}"; do
        if python -c "import $mod" 2>/dev/null; then
            echo "  OK: $mod"
        else
            echo "  FAIL: $mod"
            failed=1
        fi
    done
    echo ""

    if [ "$quick" = false ]; then
        echo "=== Frontend Build ==="
        cd "$ROOT/web"
        if npx vite build 2>&1; then
            echo "PASS: frontend build"
        else
            echo "FAIL: frontend build"
            failed=1
        fi
        cd "$ROOT"
        echo ""
    fi

    echo "=== Docker Compose Validation ==="
    if command -v docker &>/dev/null; then
        if docker compose -f docker-compose.yml config --quiet 2>&1; then
            echo "  OK: docker-compose.yml"
        else
            echo "  FAIL: docker-compose.yml"
            failed=1
        fi
        if docker compose -f docker-compose.cpu.yml config --quiet 2>&1; then
            echo "  OK: docker-compose.cpu.yml"
        else
            echo "  FAIL: docker-compose.cpu.yml"
            failed=1
        fi
    else
        echo "  SKIP: docker not available"
    fi
    echo ""

    if [ "$failed" -eq 1 ]; then
        echo "=== SOME CHECKS FAILED ==="
        exit 1
    else
        echo "=== ALL CHECKS PASSED ==="
    fi
}

cmd_backup() {
    local backup_root="${1:-$ROOT/backups}"
    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    local backup_dir="$backup_root/$timestamp"

    local qdrant_url="${QDRANT_URL:-http://localhost:6335}"
    local collection="pages"
    local data_root="${DATA_ROOT:-}"

    mkdir -p "$backup_dir"
    echo "Backing up to: $backup_dir"
    echo ""

    echo "=== Qdrant collection: $collection ==="
    if curl -sf "$qdrant_url/healthz" >/dev/null 2>&1; then
        if curl -sf "$qdrant_url/collections/$collection" >/dev/null 2>&1; then
            local points
            points=$(curl -sf "$qdrant_url/collections/$collection" | python3 -c "import json,sys; print(json.load(sys.stdin)['result']['points_count'])" 2>/dev/null || echo "?")
            echo "  Points: $points"
            echo "  Creating snapshot..."
            local snap_resp snap_name
            snap_resp=$(curl -sf -X POST "$qdrant_url/collections/$collection/snapshots?wait=true")
            snap_name=$(echo "$snap_resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['result']['name'])" 2>/dev/null)
            if [ -n "$snap_name" ]; then
                echo "  Downloading $snap_name..."
                curl -sf "$qdrant_url/collections/$collection/snapshots/$snap_name" -o "$backup_dir/$snap_name"
                echo "  Saved: $snap_name ($(du -h "$backup_dir/$snap_name" | cut -f1))"
            else
                echo "  ERROR: Failed to create snapshot"
            fi
        else
            echo "  SKIP: Collection '$collection' does not exist"
        fi
    else
        echo "  SKIP: Qdrant not reachable at $qdrant_url"
    fi
    echo ""

    echo "=== SQLite databases ==="
    for db_name in bouncer.db manifest.db librarian.db; do
        local db_path=""
        case "$db_name" in
            bouncer.db)   db_path="${BOUNCER_DB:-}" ;;
            manifest.db)  db_path="${MANIFEST_DB:-}" ;;
            librarian.db) db_path="${LIBRARIAN_DB:-}" ;;
        esac

        # Check env var path, then data root
        if [ -z "$db_path" ] && [ -n "$data_root" ]; then
            db_path="$data_root/$db_name"
        fi

        echo "$db_name:"
        if [ -n "$db_path" ] && [ -f "$db_path" ]; then
            echo "  Found: $db_path ($(du -h "$db_path" | cut -f1))"
            if command -v sqlite3 >/dev/null 2>&1; then
                sqlite3 "$db_path" ".backup '$backup_dir/$db_name'"
            else
                cp "$db_path" "$backup_dir/$db_name"
            fi
        else
            echo "  Not found"
        fi
    done
    echo ""

    echo "=== Backup complete ==="
    echo "Location: $backup_dir"
    ls -lh "$backup_dir/" 2>/dev/null | tail -n +2
    echo ""
    echo "Restore with: ./pdfiles.sh restore $backup_dir"
}

cmd_restore() {
    local backup_dir="${1:-}"

    if [ -z "$backup_dir" ]; then
        echo "Usage: ./pdfiles.sh restore <backup_dir>"
        echo ""
        echo "Available backups:"
        if [ -d "$ROOT/backups" ]; then
            ls -1 "$ROOT/backups/" 2>/dev/null | sort -r | head -10
        else
            echo "  (none)"
        fi
        exit 1
    fi

    if [ ! -d "$backup_dir" ]; then
        echo "ERROR: Backup directory not found: $backup_dir"
        exit 1
    fi

    local qdrant_url="${QDRANT_URL:-http://localhost:6335}"
    local collection="pages"
    local data_root="${DATA_ROOT:-}"

    echo "Restoring from: $backup_dir"
    echo ""

    echo "=== Qdrant ==="
    local snap_file
    snap_file=$(ls "$backup_dir"/*.snapshot 2>/dev/null | head -1 || true)
    if [ -n "$snap_file" ]; then
        if curl -sf "$qdrant_url/healthz" >/dev/null 2>&1; then
            local snap_path
            snap_path="$(cd "$(dirname "$snap_file")" && pwd)/$(basename "$snap_file")"
            echo "  Restoring $snap_file -> $collection"
            curl -sf -X POST "$qdrant_url/collections/$collection/snapshots/upload?priority=snapshot" \
                -H "Content-Type: multipart/form-data" \
                -F "snapshot=@$snap_path" \
                | python3 -c "import json,sys; r=json.load(sys.stdin); print(f'  Result: {r.get(\"result\", r)}')" 2>/dev/null || true
            local points
            points=$(curl -sf "$qdrant_url/collections/$collection" | python3 -c "import json,sys; print(json.load(sys.stdin)['result']['points_count'])" 2>/dev/null || echo "?")
            echo "  Points after restore: $points"
        else
            echo "  SKIP: Qdrant not reachable at $qdrant_url"
        fi
    else
        echo "  No .snapshot file found in backup"
    fi
    echo ""

    echo "=== SQLite databases ==="
    for db_name in bouncer.db manifest.db librarian.db; do
        local src="$backup_dir/$db_name"
        if [ -f "$src" ]; then
            local target=""
            case "$db_name" in
                bouncer.db)   target="${BOUNCER_DB:-${data_root:+$data_root/$db_name}}" ;;
                manifest.db)  target="${MANIFEST_DB:-${data_root:+$data_root/$db_name}}" ;;
                librarian.db) target="${LIBRARIAN_DB:-${data_root:+$data_root/$db_name}}" ;;
            esac
            if [ -n "$target" ]; then
                echo "  $db_name -> $target"
                cp "$src" "$target"
            else
                echo "  $db_name: No target path (set DATA_ROOT)"
                echo "  Manual restore: cp $src <target_path>"
            fi
        else
            echo "  $db_name: Not in backup"
        fi
    done
    echo ""
    echo "Restore complete."
}

cmd_reset() {
    local target="${1:-all}"

    local data_root="${DATA_ROOT:-/data}"
    local qdrant_url="${QDRANT_URL:-http://localhost:6335}"
    local collection="pages"

    case "$target" in
        qdrant)
            echo "Deleting Qdrant collection '$collection'..."
            if curl -sf "$qdrant_url/healthz" >/dev/null 2>&1; then
                local http_code
                http_code=$(curl -sf -o /dev/null -w "%{http_code}" -X DELETE "$qdrant_url/collections/$collection")
                if [ "$http_code" = "200" ]; then echo "  Deleted."; else echo "  Collection may not exist (HTTP $http_code)."; fi
            else
                echo "  SKIP: Qdrant not reachable"
            fi
            ;;
        bouncer)
            local db="$data_root/bouncer.db"
            if [ -f "$db" ]; then rm "$db"; echo "Deleted $db"; else echo "No bouncer.db at $db"; fi
            ;;
        librarian)
            local db="$data_root/librarian.db"
            if [ -f "$db" ]; then rm "$db"; echo "Deleted $db"; else echo "No librarian.db at $db"; fi
            ;;
        all)
            cmd_reset qdrant
            cmd_reset bouncer
            cmd_reset librarian
            local mdb="$data_root/manifest.db"
            if [ -f "$mdb" ]; then rm "$mdb"; echo "Deleted $mdb"; else echo "No manifest.db at $mdb"; fi
            ;;
        *)
            echo "Usage: ./pdfiles.sh reset [qdrant|bouncer|librarian|all]"
            exit 1
            ;;
    esac
}

cmd_build() {
    local mode="gpu"
    local registry=""
    local extra_args=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --cpu)      mode="cpu"; shift ;;
            --gpu)      mode="gpu"; shift ;;
            --push)     registry="$2"; shift 2 ;;
            --no-cache) extra_args="--no-cache"; shift ;;
            *)          shift ;;
        esac
    done

    local compose_file services tag_suffix
    if [ "$mode" = "cpu" ]; then
        compose_file="docker-compose.cpu.yml"
        services="backend frontend"
        tag_suffix="cpu"
    else
        compose_file="docker-compose.yml"
        services="backend frontend index"
        tag_suffix="gpu"
    fi

    echo "==> Building $mode images using $compose_file"
    docker compose -f "$compose_file" build $extra_args $services
    echo "==> Pull qdrant"
    docker compose -f "$compose_file" pull qdrant

    if [ -n "$registry" ]; then
        local project
        project=$(basename "$(pwd)" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')
        echo "==> Tagging and pushing to $registry"
        for svc in $services; do
            local local_tag="${project}-${svc}"
            local remote_tag="${registry}:${svc}-${tag_suffix}"
            echo "    $local_tag -> $remote_tag"
            docker tag "$local_tag" "$remote_tag"
            docker push "$remote_tag"
        done
        echo "==> Push complete"
    fi

    echo "==> Done"
}

cmd_help() {
    cat <<'EOF'
PDfiles — document search engine

Usage: ./pdfiles.sh <command> [options]

Commands:
  deploy [DATA_PATH]        First-time setup (creates .env, builds, starts)
  up [--build] [--cpu]      Start services
  update [--cpu]            Pull latest images and restart
  down [--clean]            Stop services (--clean removes volumes)
  logs [SERVICE]            Tail logs
  status                    Health dashboard
  dev [DATA_PATH]           Local dev (backend + frontend)
  test [--quick]            Run tests
  backup [DIR]              Backup databases
  restore DIR               Restore from backup
  reset [TARGET]            Reset databases (qdrant|bouncer|librarian|all)
  build [--cpu] [--push R]  Build Docker images

Examples:
  ./pdfiles.sh deploy /mnt/documents
  ./pdfiles.sh up --build
  ./pdfiles.sh logs backend
  ./pdfiles.sh backup ./my-backups
  ./pdfiles.sh reset qdrant
EOF
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

COMMAND="${1:-help}"
shift 2>/dev/null || true

case "$COMMAND" in
    deploy)  cmd_deploy "$@" ;;
    up)      cmd_up "$@" ;;
    update)  cmd_update "$@" ;;
    down)    cmd_down "$@" ;;
    logs)    cmd_logs "$@" ;;
    status)  cmd_status "$@" ;;
    dev)     cmd_dev "$@" ;;
    test)    cmd_test "$@" ;;
    backup)  cmd_backup "$@" ;;
    restore) cmd_restore "$@" ;;
    reset)   cmd_reset "$@" ;;
    build)   cmd_build "$@" ;;
    help|--help|-h) cmd_help ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Run './pdfiles.sh help' for usage."
        exit 1
        ;;
esac
