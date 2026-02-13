import logging
import random
import sys
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    print("Gradio is required for the UI. Install with: uv pip install 'pdfiles[dev]'")
    sys.exit(1)

from pdfiles.config import Config
from pdfiles.searcher import Searcher

logging.basicConfig(level=logging.INFO)

device = "cpu" if "--cpu" in sys.argv else "cuda:0"
cfg = Config(device=device)
searcher = Searcher(cfg)


def search_and_render(query: str, top_k: int, use_multi: bool = True) -> list[tuple]:
    """Search and return rendered page images with captions."""
    if not query.strip():
        return []

    if use_multi:
        results = searcher.search_multi(query, top_k=int(top_k))
    else:
        results = searcher.search(query, top_k=int(top_k))
    gallery_items = []

    for r in results:
        try:
            img = searcher.render_result(r)
            caption = (
                f"Score: {r.score:.4f} | {r.page_id} | "
                f"Page {r.page_index + 1}/{r.total_pages} of {r.pdf_id}"
            )
            gallery_items.append((img, caption))
        except Exception as e:
            logging.error("Failed to render %s: %s", r.page_id, e)

    return gallery_items


def _load_librarian_db():
    """Load LibrarianDB if the database file exists."""
    from pdfiles.librarian import LibrarianDB

    if not cfg.librarian_db.exists():
        return None
    return LibrarianDB(cfg.librarian_db)


# ---------------------------------------------------------------------------
# Shelf browsing
# ---------------------------------------------------------------------------

def get_shelf_data() -> list[dict]:
    """Get all shelves for display."""
    db = _load_librarian_db()
    if db is None:
        return []
    shelves = db.get_all_shelves()
    db.close()
    return shelves


def render_shelf_pages(page_ids: list[str]) -> list[tuple]:
    """Render page thumbnails for a list of page IDs."""
    from pdfiles.qdrant_store import QdrantStore
    from pdfiles.renderer import render_page

    store = QdrantStore(cfg)
    gallery_items = []

    for page_id in page_ids:
        try:
            point_id = int(page_id)
            points = store.client.retrieve(
                collection_name=store.collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                continue
            payload = points[0].payload
            img = render_page(
                cfg.resolve_pdf_path(payload["pdf_path"]),
                payload["page_index"],
                dpi=cfg.render_dpi,
            )
            caption = (
                f"{page_id} | "
                f"Page {payload['page_index'] + 1}/{payload['total_pages']} of {payload['pdf_id']}"
            )
            gallery_items.append((img, caption))
        except Exception as e:
            logging.error("Failed to render %s: %s", page_id, e)

    return gallery_items


def build_shelf_html(shelves: list[dict]) -> str:
    """Build HTML for the shelves overview grid."""
    if not shelves:
        return "<p>No shelves found. Run <code>pdfiles librarian run</code> first.</p>"

    html_parts = ['<div style="display:flex;flex-wrap:wrap;gap:12px;">']
    for s in shelves:
        count = s["page_count"]
        mean_z = s["mean_z_score"]
        cat = s["category"]
        html_parts.append(f"""
        <div style="border:1px solid #444;border-radius:8px;padding:12px;width:220px;
                    background:#1a1a2e;cursor:pointer;" class="shelf-card">
            <div style="font-weight:bold;font-size:14px;margin-bottom:4px;">{cat}</div>
            <div style="color:#aaa;font-size:12px;">{count:,} pages &middot; Z={mean_z:.1f}</div>
        </div>
        """)
    html_parts.append("</div>")
    return "".join(html_parts)


def get_shelf_choices() -> list[str]:
    """Get dropdown choices for shelf selection."""
    shelves = get_shelf_data()
    if not shelves:
        return ["No shelves found - run 'pdfiles librarian run' first"]
    return [
        f"{s['category']} ({s['page_count']:,} pages, Z={s['mean_z_score']:.1f})"
        for s in shelves
    ]


def browse_shelf(shelf_choice: str) -> list[tuple]:
    """Render pages for a selected shelf."""
    if not shelf_choice or "No shelves" in shelf_choice:
        return []

    # Parse category name from "category (N pages, Z=X.X)"
    category = shelf_choice.rsplit(" (", 1)[0]

    db = _load_librarian_db()
    if db is None:
        return []

    pages = db.get_shelf_pages(category, limit=50)
    db.close()

    if not pages:
        return []

    page_ids = [p["page_id"] for p in pages]
    return render_shelf_pages(page_ids)


def refresh_unsorted() -> list[tuple]:
    """Render a fresh random sample from the unsorted shelf."""
    db = _load_librarian_db()
    if db is None:
        return []

    pages = db.get_shelf_pages("Unsorted / Random Discovery", limit=50)
    db.close()

    if not pages:
        return []

    # Shuffle for discovery
    random.shuffle(pages)
    page_ids = [p["page_id"] for p in pages[:20]]
    return render_shelf_pages(page_ids)


# ---------------------------------------------------------------------------
# Legacy cluster browsing (kept for backwards compat)
# ---------------------------------------------------------------------------

def get_cluster_choices() -> list[str]:
    """Get dropdown choices for cluster selection."""
    db = _load_librarian_db()
    if db is None:
        return ["No clusters found - run 'pdfiles librarian run' first"]
    clusters = db.get_all_clusters()
    db.close()
    if not clusters:
        return ["No clusters found - run 'pdfiles librarian run' first"]
    return [
        f"Cluster {c['cluster_id']}: {c['label']} ({c['page_count']} pages)"
        for c in clusters
    ]


def browse_cluster(cluster_choice: str) -> list[tuple]:
    """Render representative pages for a selected cluster."""
    if not cluster_choice or "No clusters" in cluster_choice:
        return []

    try:
        cluster_id = int(cluster_choice.split(":")[0].replace("Cluster ", ""))
    except (ValueError, IndexError):
        return []

    db = _load_librarian_db()
    if db is None:
        return []

    clusters = db.get_all_clusters()
    representative_ids = []
    for c in clusters:
        if c["cluster_id"] == cluster_id:
            representative_ids = c["representative_ids"]
            break

    if not representative_ids:
        pages = db.get_cluster_pages(cluster_id)
        representative_ids = [pid for pid, _ in pages[:10]]

    db.close()
    return render_shelf_pages(representative_ids)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="PDfiles") as demo:
    gr.Markdown("# PDfiles")
    gr.Markdown("Search through litigation documents using natural language queries.")

    with gr.Tab("Search"):
        with gr.Row():
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="e.g., handwritten notes, financial statements, blueprints...",
                scale=4,
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=50, value=10, step=1,
                label="Results",
                scale=1,
            )
            multi_query_toggle = gr.Checkbox(
                label="Multi-query RRF",
                value=True,
                info="Better quality, 3x slower",
            )

        search_btn = gr.Button("Search", variant="primary")

        gallery = gr.Gallery(
            label="Results",
            columns=2,
            height="auto",
            object_fit="contain",
        )

        search_btn.click(
            fn=search_and_render,
            inputs=[query_input, top_k_slider, multi_query_toggle],
            outputs=gallery,
        )
        query_input.submit(
            fn=search_and_render,
            inputs=[query_input, top_k_slider, multi_query_toggle],
            outputs=gallery,
        )

    with gr.Tab("Browse Categories"):
        gr.Markdown("Browse documents organized by category. Categories ranked by match strength.")
        shelf_dropdown = gr.Dropdown(
            choices=get_shelf_choices(),
            label="Select Category",
            interactive=True,
        )
        with gr.Row():
            refresh_shelves_btn = gr.Button("Refresh Categories")
            refresh_unsorted_btn = gr.Button("Random Discovery")

        shelf_gallery = gr.Gallery(
            label="Category Pages",
            columns=3,
            height="auto",
            object_fit="contain",
        )

        shelf_dropdown.change(
            fn=browse_shelf,
            inputs=[shelf_dropdown],
            outputs=shelf_gallery,
        )
        refresh_shelves_btn.click(
            fn=lambda: gr.update(choices=get_shelf_choices()),
            outputs=shelf_dropdown,
        )
        refresh_unsorted_btn.click(
            fn=refresh_unsorted,
            outputs=shelf_gallery,
        )


def main():
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
