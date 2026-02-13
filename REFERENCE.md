# PDfiles — Reference

## Docker Deployment

```bash
# Quick start — pull pre-built images
docker compose pull
docker compose up -d

# Or build from source
docker compose up -d --build

# CPU-only mode (no GPU required)
docker compose -f docker-compose.cpu.yml pull
docker compose -f docker-compose.cpu.yml up -d
```

Configure via `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | *(required)* | Path to your PDF documents |
| `WEB_PORT` | 80 | Web UI port |
| `ADMIN_MODE` | false | Enable indexing/export features |
| `BACKEND_DEVICE` | cuda:0 | GPU device for backend |
| `INDEX_DEVICE` | cuda:0 | GPU device for indexing |

### Management Script

```bash
./pdfiles.sh deploy /mnt/documents    # First-time setup
./pdfiles.sh up [--build] [--cpu]     # Start services
./pdfiles.sh down [--clean]           # Stop (--clean removes volumes)
./pdfiles.sh logs [SERVICE]           # Tail logs
./pdfiles.sh status                   # Health dashboard
./pdfiles.sh backup [DIR]             # Backup databases
./pdfiles.sh restore DIR              # Restore from backup
./pdfiles.sh reset [TARGET]           # Reset (qdrant|bouncer|librarian|all)
```

## CLI Commands

### Indexing

```bash
pdfiles index                                # Index all pages
pdfiles index --folder 0330                  # Index one subfolder
pdfiles index --filter VISUAL                # Only bouncer-approved pages
pdfiles index --batch-size 4 --dpi 200       # Tuning params
pdfiles index --limit 100                    # Test with N pages
pdfiles status                               # Show indexing progress
```

### Search

```bash
pdfiles search "surveillance photographs"    # Text search
pdfiles search "handwritten notes" --top-k 20
```

### Bouncer (Page Classification)

```bash
pdfiles bouncer run                          # Classify all pages (Tier 1 + Tier 2)
pdfiles bouncer run --no-tier2               # Tier 1 only (fast, CPU)
pdfiles bouncer run --folder 0330            # One subfolder
pdfiles bouncer run --limit 500             # Test with N pages
pdfiles bouncer stats                        # Show VISUAL / TEXT_ONLY / UNCERTAIN counts
pdfiles bouncer sample --n 500              # Sample text_ratio distribution
```

### Librarian (Clustering)

```bash
pdfiles librarian run                        # Cluster with k=30 (default)
pdfiles librarian run --k 20                 # Custom cluster count
pdfiles librarian run --device cpu           # Force CPU for label embedding
pdfiles librarian stats                      # Show cluster sizes + labels
pdfiles librarian show 5                     # List pages in cluster 5
pdfiles librarian show 5 --limit 50          # Show more pages
```

### Web UI

```bash
uv run python -m pdfiles.app          # GPU mode (cuda:0)
uv run python -m pdfiles.app --cpu    # CPU mode
# Opens at http://localhost:7860
# Tabs: Search | Browse Clusters
```

## Testing

```bash
uv run pytest tests/ -v                                   # All tests (needs Qdrant running)
uv run pytest tests/ -v --ignore=tests/test_qdrant_store.py  # Skip Qdrant tests
uv run pytest tests/test_librarian.py -v                  # Librarian only
uv run pytest tests/test_bouncer.py -v                    # Bouncer only
```


## Key Files

| File | Purpose |
|------|---------|
| `src/pdfiles/config.py` | All configuration (paths, thresholds, model settings) |
| `src/pdfiles/opt_parser.py` | Parse VOL00011.OPT file into PageRecord list |
| `src/pdfiles/renderer.py` | PDF page to PIL Image (PyMuPDF) |
| `src/pdfiles/embedder.py` | ColQwen2.5 model wrapper (image + query embedding) |
| `src/pdfiles/pooling.py` | Multi-vector pooling (1030 -> 262 vectors) |
| `src/pdfiles/qdrant_store.py` | Qdrant client (upsert, search, export vectors) |
| `src/pdfiles/indexer.py` | Orchestrates render -> embed -> pool -> upsert pipeline |
| `src/pdfiles/searcher.py` | Query expansion + search + result rendering |
| `src/pdfiles/bouncer.py` | Two-tier page classification (text_ratio + Surya layout) |
| `src/pdfiles/librarian.py` | K-Means clustering + auto-labeling (text + reverse search) |
| `src/pdfiles/manifest.py` | SQLite manifest DB for page/PDF metadata |
| `src/pdfiles/pipeline.py` | Unified page record loading (OPT + manifest) |
| `src/pdfiles/api.py` | FastAPI REST backend (search, status, page images) |
| `src/pdfiles/cli.py` | Click CLI (index, search, bouncer, librarian, status) |
| `src/pdfiles/app.py` | Gradio web UI (Search + Browse Clusters tabs) |

## Data Layout

```
VOL00011/
  DATA/VOL00011.OPT          # Page manifest (ID, path, page_count)
  IMAGES/0330/.../*.pdf       # Source PDFs
  bouncer.db                  # Page classifications (SQLite)
  librarian.db                # Cluster assignments + labels (SQLite)
```

## Configuration Defaults

| Setting | Default | Description |
|---------|---------|-------------|
| `render_dpi` | 200 | PDF rendering resolution |
| `batch_size` | 4 | Embedding batch size |
| `device` | cuda:0 | Model device |
| `num_clusters` | 30 | K-Means cluster count |
| `bouncer_high_threshold` | 0.30 | Text ratio above = TEXT_ONLY |
| `bouncer_low_threshold` | 0.05 | Text ratio below = VISUAL |
| `vector_dim` | 128 | ColQwen2.5 embedding dimension |
| `pooled_vectors` | 262 | Vectors per page after pooling |
