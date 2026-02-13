import logging

import click

from pdfiles.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@click.group()
def cli():
    """PDfiles document search."""
    pass


@cli.command()
@click.option("--batch-size", default=4, help="Batch size for embedding")
@click.option("--dpi", default=200, help="Render DPI")
@click.option("--limit", default=None, type=int, help="Max pages to index (for testing)")
@click.option("--filter", "filter_cls", default=None, type=click.Choice(["VISUAL", "TEXT_ONLY"]),
              help="Only index pages with this bouncer classification")
@click.option("--folder", default=None, help="Only index pages from this IMAGES subfolder (e.g. 0330)")
def index(batch_size: int, dpi: int, limit: int | None, filter_cls: str | None, folder: str | None):
    """Index PDF pages into Qdrant."""
    from pdfiles.indexer import run_indexing

    cfg = Config(batch_size=batch_size, render_dpi=dpi)
    run_indexing(cfg, limit=limit, filter_classification=filter_cls, folder=folder)


@cli.command()
@click.argument("query")
@click.option("--top-k", default=10, help="Number of results")
def search(query: str, top_k: int):
    """Search for pages matching a text query."""
    from pdfiles.searcher import Searcher

    cfg = Config()
    searcher = Searcher(cfg)
    results = searcher.search(query, top_k=top_k)

    for i, r in enumerate(results, 1):
        click.echo(
            f"{i:3d}. [{r.score:.4f}] {r.page_id} "
            f"(page {r.page_index + 1}/{r.total_pages} of {r.pdf_id})"
        )


@cli.command()
@click.option("--filter", "filter_cls", default="VISUAL",
              type=click.Choice(["VISUAL", "TEXT_ONLY"]),
              help="Only index pages with this bouncer classification")
@click.option("--batch-size", default=4, help="Batch size for embedding")
def pipeline(filter_cls: str, batch_size: int):
    """Run streaming scan -> classify -> index pipeline."""
    from pdfiles.embedder import Embedder
    from pdfiles.pipeline import run_streaming_pipeline

    cfg = Config(batch_size=batch_size)
    click.echo(f"Loading embedder on {cfg.device}...")
    embedder = Embedder(cfg)

    def _progress(state: dict):
        indexed = state.get("indexed", 0)
        total = state.get("total", 0)
        scanned = state.get("scanned_pdfs", 0)
        classified = state.get("classified_pages", 0)
        errors = state.get("errors", 0)
        click.echo(
            f"\rScan: {scanned} PDFs | Classify: {classified} | "
            f"Index: {indexed}/{total} | Errors: {errors}",
            nl=False,
        )

    click.echo("Starting streaming pipeline...")
    run_streaming_pipeline(
        cfg=cfg,
        embedder=embedder,
        progress_callback=_progress,
        exclude_classification=filter_cls,
    )
    click.echo("\nPipeline complete.")


@cli.command()
def status():
    """Show indexing status."""
    from pdfiles.opt_parser import load_page_records
    from pdfiles.qdrant_store import QdrantStore

    cfg = Config()
    store = QdrantStore(cfg)
    try:
        count = store.count()
        click.echo(f"Collection: {cfg.collection_name}")
        click.echo(f"Indexed pages: {count:,}")
        try:
            records = load_page_records(cfg)
            target = len(records)
            click.echo(f"Target pages: {target:,}")
            click.echo(f"Progress: {count / target * 100:.1f}%")
        except Exception:
            click.echo(f"Target pages: unknown (no OPT file or data directory)")
    except Exception as e:
        click.echo(f"Error: {e}")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8000, type=int, help="Bind port")
def serve(host: str, port: int):
    """Start the FastAPI API server."""
    import uvicorn

    uvicorn.run("pdfiles.api:app", host=host, port=port)


@cli.command("export-index")
@click.argument("output_dir", type=click.Path())
def export_index(output_dir: str):
    """Export Qdrant collection snapshot for portability."""
    from pathlib import Path
    from urllib.request import urlretrieve

    from pdfiles.qdrant_store import QdrantStore

    cfg = Config()
    store = QdrantStore(cfg)

    click.echo(f"Creating snapshot of '{cfg.collection_name}'...")
    snapshot = store.client.create_snapshot(cfg.collection_name, wait=True)
    click.echo(f"Snapshot created: {snapshot.name}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    url = f"{cfg.qdrant_url}/collections/{cfg.collection_name}/snapshots/{snapshot.name}"
    dest = out / snapshot.name
    click.echo(f"Downloading to {dest}...")
    urlretrieve(url, str(dest))
    click.echo(f"Snapshot saved: {dest}")


@cli.command("import-index")
@click.argument("snapshot_path", type=click.Path(exists=True))
def import_index(snapshot_path: str):
    """Restore Qdrant collection from a snapshot file."""
    from pathlib import Path

    from pdfiles.qdrant_store import QdrantStore

    cfg = Config()
    store = QdrantStore(cfg)
    snap = Path(snapshot_path).resolve()

    click.echo(f"Recovering '{cfg.collection_name}' from {snap}...")
    store.client.recover_snapshot(
        cfg.collection_name,
        location=f"file://{snap}",
        wait=True,
    )
    count = store.count()
    click.echo(f"Recovery complete. Collection has {count:,} points.")


if __name__ == "__main__":
    cli()
