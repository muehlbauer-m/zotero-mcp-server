#!/usr/bin/env python3
"""
Zotero MCP Server - Model Context Protocol server for Zotero integration.

This server provides tools and resources for interacting with Zotero libraries,
using the latest MCP paradigms and the official Python SDK.
"""

import os
import sys
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from dotenv import load_dotenv
from pyzotero import zotero
from mcp.server.fastmcp import FastMCP
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger('zotero-mcp-server')

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("Zotero MCP Server")

# Global Zotero client instance
zot: Optional[zotero.Zotero] = None

# Semantic search globals
_semantic_index: Optional[dict] = None
_embedding_model = None
INDEX_DIR = Path.home() / ".zotero-mcp"
INDEX_FILE = INDEX_DIR / "semantic_index.pkl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def init_zotero_client():
    """Initialize the Zotero client with credentials from environment."""
    global zot

    api_key = os.getenv('ZOTERO_API_KEY')
    user_id = os.getenv('ZOTERO_USER_ID')
    group_id = os.getenv('ZOTERO_GROUP_ID')

    if not api_key:
        logger.error("ZOTERO_API_KEY environment variable not set")
        return

    try:
        # Prioritize user library over group library
        if user_id:
            zot = zotero.Zotero(user_id, 'user', api_key)
            logger.info(f"Initialized Zotero client for user {user_id}")
        elif group_id:
            zot = zotero.Zotero(group_id, 'group', api_key)
            logger.info(f"Initialized Zotero client for group {group_id}")
        else:
            logger.error("Either ZOTERO_USER_ID or ZOTERO_GROUP_ID must be set")
    except Exception as e:
        logger.error(f"Error initializing Zotero client: {str(e)}")


def ensure_client():
    """Ensure Zotero client is initialized."""
    if zot is None:
        raise RuntimeError("Zotero client not initialized. Check API credentials.")


def get_embedding_model():
    """Lazy-load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def load_semantic_index() -> dict:
    """Load the semantic index from disk, or return empty structure."""
    global _semantic_index
    if _semantic_index is not None:
        return _semantic_index

    if INDEX_FILE.exists():
        try:
            with open(INDEX_FILE, 'rb') as f:
                _semantic_index = pickle.load(f)
            logger.info(f"Loaded semantic index with {len(_semantic_index.get('items', {}))} items")
        except Exception as e:
            logger.error(f"Failed to load semantic index: {e}")
            _semantic_index = {"model": EMBEDDING_MODEL_NAME, "chunk_size": CHUNK_SIZE,
                             "chunk_overlap": CHUNK_OVERLAP, "created": None, "items": {}}
    else:
        _semantic_index = {"model": EMBEDDING_MODEL_NAME, "chunk_size": CHUNK_SIZE,
                          "chunk_overlap": CHUNK_OVERLAP, "created": None, "items": {}}
    return _semantic_index


def save_semantic_index(index: dict):
    """Save the semantic index to disk."""
    global _semantic_index
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(index, f)
    _semantic_index = index
    logger.info(f"Saved semantic index with {len(index.get('items', {}))} items")


def extract_pdf_text_with_pages(item_key: str) -> Optional[list[dict]]:
    """
    Extract text from PDF with page tracking.
    Returns list of {"page": N, "text": "..."} or None if no PDF.
    """
    import requests
    import fitz  # pymupdf

    try:
        # Get item metadata from local API
        response = requests.get(
            f"http://localhost:23119/api/users/0/items/{item_key}",
            timeout=10
        )
        response.raise_for_status()
        item_data = response.json()

        item_type = item_data.get("data", {}).get("itemType", "")

        if item_type == "attachment":
            pdf_path = item_data.get("data", {}).get("path", "")
        else:
            # Find PDF attachment
            children_response = requests.get(
                f"http://localhost:23119/api/users/0/items/{item_key}/children",
                timeout=10
            )
            children_response.raise_for_status()
            children = children_response.json()

            pdf_path = None
            for child in children:
                if child.get("data", {}).get("contentType") == "application/pdf":
                    pdf_path = child.get("data", {}).get("path", "")
                    break

        if not pdf_path:
            return None

        # Extract text page by page
        doc = fitz.open(pdf_path)
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append({"page": page_num + 1, "text": text})
        doc.close()

        return pages if pages else None

    except Exception as e:
        logger.debug(f"Failed to extract PDF for {item_key}: {e}")
        return None


def chunk_text_with_pages(pages: list[dict], title: str, abstract: str,
                          chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping chunks, tracking source page.
    Returns list of {"text": "...", "page": N, "char_start": N}
    """
    # Prepend title (2x for weighting) and abstract
    header = f"{title}. {title}. {abstract or ''}\n\n"

    # Build full text with page boundaries
    full_text = header
    page_boundaries = [(0, len(header), 0)]  # (start, end, page_num) - page 0 for header

    for page_data in pages:
        start = len(full_text)
        full_text += page_data["text"] + "\n\n"
        page_boundaries.append((start, len(full_text), page_data["page"]))

    # Chunk the text
    chunks = []
    pos = 0
    while pos < len(full_text):
        chunk_end = min(pos + chunk_size, len(full_text))
        chunk_text = full_text[pos:chunk_end].strip()

        if chunk_text:
            # Find which page this chunk is primarily from
            chunk_mid = pos + len(chunk_text) // 2
            chunk_page = 0
            for start, end, page in page_boundaries:
                if start <= chunk_mid < end:
                    chunk_page = page
                    break

            chunks.append({
                "text": chunk_text,
                "page": chunk_page,
                "char_start": pos
            })

        # Move forward with overlap
        pos += chunk_size - overlap
        if pos >= len(full_text) - overlap:
            break

    return chunks


# ============================================================================
# RESOURCES - Read-only data access
# ============================================================================

@mcp.resource("zotero://collections")
def get_collections() -> str:
    """List of collections in the Zotero library."""
    ensure_client()
    collections = zot.collections()
    return json.dumps(collections, indent=2)


@mcp.resource("zotero://items/top")
def get_top_items() -> str:
    """Top-level items in the Zotero library."""
    ensure_client()
    items = zot.top(limit=50)
    return json.dumps(items, indent=2)


@mcp.resource("zotero://items/recent")
def get_recent_items() -> str:
    """Recently added or modified items in the Zotero library."""
    ensure_client()
    items = zot.items(limit=20, sort="dateModified", direction="desc")
    return json.dumps(items, indent=2)


@mcp.resource("zotero://collections/{collection_key}/items")
def get_collection_items(collection_key: str) -> str:
    """Items in a specific Zotero collection."""
    ensure_client()
    items = zot.collection_items(collection_key)
    return json.dumps(items, indent=2)


@mcp.resource("zotero://items/{item_key}")
def get_item(item_key: str) -> str:
    """Details of a specific Zotero item."""
    ensure_client()
    item = zot.item(item_key)
    return json.dumps(item, indent=2)


@mcp.resource("zotero://items/{item_key}/citation/{style}")
def get_item_citation(item_key: str, style: str) -> str:
    """Citation for a specific Zotero item in a specific style."""
    ensure_client()
    citation = zot.item(item_key, format="citation", style=style)
    return citation


# ============================================================================
# TOOLS - Actions and operations
# ============================================================================

@mcp.tool()
def search_items(
    query: str,
    collection_key: Optional[str] = None,
    limit: int = 20
) -> str:
    """
    Search for items in the Zotero library.

    Args:
        query: Search query string (use empty string "" to list all items)
        collection_key: Optional collection key to search/list within
        limit: Maximum number of results to return (default: 20)

    Returns:
        JSON string containing search results
    """
    ensure_client()

    if collection_key:
        if query and query != "*":
            # Search within collection
            search_params = {"q": query, "limit": limit}
            items = zot.collection_items_top(collection_key, **search_params)
        else:
            # List all items in collection (no search query)
            items = zot.collection_items(collection_key, limit=limit)
    else:
        if query and query != "*":
            # Search entire library
            search_params = {"q": query, "limit": limit}
            items = zot.items(**search_params)
        else:
            # List recent items (no search query)
            items = zot.items(limit=limit)

    result = {
        "query": query,
        "collection_key": collection_key,
        "count": len(items),
        "results": items
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def get_item(item_key: str) -> str:
    """
    Get full details of a specific item by its key.

    Args:
        item_key: The Zotero item key

    Returns:
        JSON string containing item data
    """
    ensure_client()
    item = zot.item(item_key)
    return json.dumps(item, indent=2)


@mcp.tool()
def get_citation(item_key: str, style: str = "apa") -> str:
    """
    Get citation for a specific item.

    Args:
        item_key: The Zotero item key
        style: Citation style (e.g., apa, mla, chicago). Default: apa

    Returns:
        Formatted citation string
    """
    ensure_client()
    citation = zot.item(item_key, format="citation", style=style)
    return citation


@mcp.tool()
def add_item(
    item_type: str,
    title: str,
    creators: Optional[list[dict[str, str]]] = None,
    collection_key: Optional[str] = None,
    additional_fields: Optional[dict[str, Any]] = None
) -> str:
    """
    Add a new item to the Zotero library.

    Args:
        item_type: Item type (e.g., journalArticle, book, webpage)
        title: Item title
        creators: List of creators with format [{"creatorType": "author", "firstName": "...", "lastName": "..."}]
        collection_key: Optional collection key to add the item to
        additional_fields: Additional fields (e.g., date, url, publisher)

    Returns:
        JSON string with creation response
    """
    ensure_client()

    # Create item template
    template = zot.item_template(item_type)

    # Set title
    template["title"] = title

    # Set creators
    if creators:
        template["creators"] = creators

    # Set additional fields
    if additional_fields:
        for key, value in additional_fields.items():
            template[key] = value

    # Create item
    response = zot.create_items([template])

    # Add to collection if specified
    if collection_key and response.get("success"):
        item_key = response["successful"]["0"]["key"]
        zot.addto_collection(collection_key, [item_key])

    return json.dumps(response, indent=2)


@mcp.tool()
def get_bibliography(item_keys: list[str], style: str = "apa") -> str:
    """
    Get bibliography for multiple items.

    Args:
        item_keys: List of Zotero item keys
        style: Citation style (e.g., apa, mla, chicago). Default: apa

    Returns:
        Formatted bibliography string
    """
    ensure_client()
    bibliography = zot.bibliography(item_keys, style=style)
    return bibliography


@mcp.tool()
def create_collection(name: str, parent_key: Optional[str] = None) -> str:
    """
    Create a new collection in the Zotero library.

    Args:
        name: Name of the new collection
        parent_key: Optional parent collection key for nested collections

    Returns:
        JSON string with creation response
    """
    ensure_client()

    collection_data = {"name": name}
    if parent_key:
        collection_data["parentCollection"] = parent_key

    response = zot.create_collections([collection_data])
    return json.dumps(response, indent=2)


@mcp.tool()
def update_item(
    item_key: str,
    updates: dict[str, Any]
) -> str:
    """
    Update an existing item in the Zotero library.

    Args:
        item_key: The Zotero item key to update
        updates: Dictionary of fields to update

    Returns:
        JSON string with update response
    """
    ensure_client()

    # Get the existing item
    item = zot.item(item_key)

    # Extract just the data portion (API returns library, meta, links, data)
    item_data = item.get('data', item)

    # Update fields in data
    for key, value in updates.items():
        item_data[key] = value

    # Update the item (pyzotero expects the data dict)
    response = zot.update_item(item_data)
    return json.dumps(response, indent=2)


@mcp.tool()
def delete_item(item_key: str) -> str:
    """
    Delete an item from the Zotero library.

    Args:
        item_key: The Zotero item key to delete

    Returns:
        Success message
    """
    ensure_client()

    # Get item version for deletion
    item = zot.item(item_key)
    version = item.get('version')

    # Delete the item
    zot.delete_item(item, version=version)

    return json.dumps({"success": True, "message": f"Item {item_key} deleted"})


@mcp.tool()
def get_item_types() -> str:
    """
    Get list of all available Zotero item types.

    Returns:
        JSON string containing all item types
    """
    ensure_client()
    item_types = zot.item_types()
    return json.dumps(item_types, indent=2)


@mcp.tool()
def get_item_fields(item_type: str) -> str:
    """
    Get available fields for a specific item type.

    Args:
        item_type: The item type to get fields for (e.g., journalArticle, book)

    Returns:
        JSON string containing available fields
    """
    ensure_client()
    fields = zot.item_type_fields(item_type)
    return json.dumps(fields, indent=2)


@mcp.tool()
def get_fulltext_local(item_key: str) -> str:
    """
    Get full text content from local Zotero desktop (requires Zotero running).

    Works with ZotMoov-managed files since Zotero indexes PDFs locally.

    Args:
        item_key: The Zotero item key

    Returns:
        Full text content or error message if not available
    """
    import requests
    import fitz  # pymupdf

    try:
        # Step 1: Get item metadata from local API to find PDF path
        response = requests.get(
            f"http://localhost:23119/api/users/0/items/{item_key}",
            timeout=10
        )
        response.raise_for_status()
        item_data = response.json()

        # Check if this is an attachment or parent item
        item_type = item_data.get("data", {}).get("itemType", "")

        if item_type == "attachment":
            # This is the attachment itself
            pdf_path = item_data.get("data", {}).get("path", "")
        else:
            # This is a parent item, need to find child attachments
            children_response = requests.get(
                f"http://localhost:23119/api/users/0/items/{item_key}/children",
                timeout=10
            )
            children_response.raise_for_status()
            children = children_response.json()

            # Find PDF attachment
            pdf_path = None
            for child in children:
                if child.get("data", {}).get("contentType") == "application/pdf":
                    pdf_path = child.get("data", {}).get("path", "")
                    break

            if not pdf_path:
                return json.dumps({
                    "error": "No PDF attachment found for this item",
                    "item_key": item_key
                }, indent=2)

        if not pdf_path:
            return json.dumps({
                "error": "No file path found in attachment metadata",
                "item_key": item_key
            }, indent=2)

        # Step 2: Read PDF and extract text
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            for page_num, page in enumerate(doc):
                text_content.append(f"--- Page {page_num + 1} ---\n{page.get_text()}")
            doc.close()

            full_text = "\n\n".join(text_content)

            return json.dumps({
                "item_key": item_key,
                "pdf_path": pdf_path,
                "page_count": len(text_content),
                "content": full_text
            }, indent=2)

        except Exception as pdf_error:
            return json.dumps({
                "error": f"Failed to read PDF: {str(pdf_error)}",
                "pdf_path": pdf_path,
                "note": "Ensure the file exists and is accessible"
            }, indent=2)

    except requests.exceptions.ConnectionError:
        return json.dumps({
            "error": "Cannot connect to Zotero local API",
            "note": "Ensure Zotero desktop is running with local API enabled (Edit > Settings > Advanced > Allow other applications to communicate with Zotero)"
        }, indent=2)
    except requests.exceptions.HTTPError as e:
        return json.dumps({
            "error": f"HTTP error: {e.response.status_code}",
            "note": "Item not found or API error"
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def list_collections(parent_key: Optional[str] = None) -> str:
    """
    List all collections in the Zotero library.

    Args:
        parent_key: Optional parent collection key to list only subcollections

    Returns:
        JSON string containing collections with their keys, names, and hierarchy
    """
    ensure_client()

    if parent_key:
        collections = zot.collections_sub(parent_key)
    else:
        collections = zot.collections()

    # Simplify output for readability
    result = []
    for col in collections:
        data = col.get('data', col)
        result.append({
            "key": data.get('key'),
            "name": data.get('name'),
            "parentCollection": data.get('parentCollection', False),
            "numItems": col.get('meta', {}).get('numItems', 0)
        })

    return json.dumps(result, indent=2)


# ============================================================================
# SEMANTIC SEARCH TOOLS
# ============================================================================

class ProgressWindow:
    """Simple progress window for long-running operations."""
    def __init__(self, title="Building Index"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("500x200")
        self.root.resizable(False, False)

        # Title
        title_label = tk.Label(self.root, text=title, font=("Arial", 12, "bold"))
        title_label.pack(pady=10)

        # Status text
        self.status_label = tk.Label(self.root, text="Initializing...", wraplength=480)
        self.status_label.pack(pady=5)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.root, variable=self.progress_var, maximum=100, mode='determinate'
        )
        self.progress_bar.pack(pady=10, padx=20, fill=tk.X)

        # Details text
        self.details_label = tk.Label(self.root, text="", fg="gray", wraplength=480)
        self.details_label.pack(pady=5)

        # Make window stay on top
        self.root.attributes('-topmost', True)

        # Don't block - run in background
        self.root.update_idletasks()

    def update(self, status: str, progress: float = None, details: str = ""):
        """Update progress window."""
        try:
            self.status_label.config(text=status)
            if details:
                self.details_label.config(text=details)
            if progress is not None:
                self.progress_var.set(progress)
            self.root.update_idletasks()
        except Exception as e:
            logger.error(f"Error updating progress window: {e}")

    def close(self):
        """Close the progress window."""
        try:
            self.root.destroy()
        except Exception as e:
            logger.error(f"Error closing progress window: {e}")


@mcp.tool()
def build_semantic_index(force_rebuild: bool = False) -> str:
    """
    Build or update the semantic search index from local Zotero library.

    Extracts text from all PDFs, splits into chunks, and creates embeddings.
    Requires Zotero desktop to be running with local API enabled.

    This operation can take 5-30 minutes depending on library size and PDF count.
    - First run: Downloads ~80MB embedding model
    - PDF extraction: ~1-5 min per 100 PDFs
    - Embedding: ~5-10 min per 1000 chunks

    Args:
        force_rebuild: If True, rebuild entire index. If False, only update changed items.

    Returns:
        JSON with indexing statistics
    """
    import requests
    import time

    start_time = time.time()
    stats = {"items_processed": 0, "items_skipped": 0, "items_failed": 0,
             "chunks_created": 0, "errors": [], "status_messages": []}

    # Create progress window
    try:
        progress_win = ProgressWindow("Building Semantic Index")
        progress_win.update("Fetching items from Zotero...", 0)
    except Exception as e:
        logger.warning(f"Could not create progress window: {e}")
        progress_win = None

    try:
        # Get all items from local Zotero API
        # Local API has a ~40 item limit per request
        response = requests.get(
            "http://localhost:23119/api/users/0/items?limit=40",
            timeout=30
        )
        response.raise_for_status()
        all_items = response.json()

        logger.info(f"Fetched {len(all_items)} items from Zotero")

        # Check if there are more items via Total-Results header
        total_results = int(response.headers.get("Total-Results", len(all_items)))
        if len(all_items) < total_results:
            logger.warning(f"Only fetched {len(all_items)} of {total_results} items. "
                          "Zotero local API pagination is limited.")
    except requests.exceptions.ConnectionError:
        return json.dumps({
            "error": "Cannot connect to Zotero local API",
            "note": "Ensure Zotero desktop is running with local API enabled"
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

    # Filter to parent items only (not attachments or notes)
    parent_items = [item for item in all_items
                    if item.get("data", {}).get("itemType") not in
                    ("attachment", "note", "annotation")]

    logger.info(f"Found {len(parent_items)} parent items to index")

    # Load existing index
    index = load_semantic_index()
    if force_rebuild:
        index["items"] = {}

    # Get embedding model (this may download ~80MB on first run)
    logger.info("Loading embedding model (may take 1-2 minutes on first run)...")
    if progress_win:
        progress_win.update("Loading embedding model (~80MB on first run)...", 5,
                           "This may take 1-2 minutes the first time")
    model = get_embedding_model()
    if progress_win:
        progress_win.update("Model loaded! Starting PDF extraction...", 10)
    logger.info("Embedding model loaded successfully")

    # Track current item keys for cleanup
    current_keys = set()
    total_to_process = len([i for i in parent_items
                           if not (index["items"].get(i.get("key"), {}).get("version") == i.get("version")
                                  and not force_rebuild)])

    # Track PDF extraction progress
    pdfs_extracted = 0
    pdfs_with_text = 0

    for idx, item in enumerate(parent_items):
        item_key = item.get("key")
        item_data = item.get("data", {})
        item_version = item.get("version", 0)
        title = item_data.get("title", "Untitled")[:50]  # Truncate for display

        current_keys.add(item_key)

        # Update progress
        progress_pct = 10 + (idx / len(parent_items)) * 85  # 10-95% for processing
        if progress_win:
            progress_win.update(
                f"Processing: {idx + 1}/{len(parent_items)} items",
                progress_pct,
                f"{title}..."
            )

        # Check if item needs updating
        existing = index["items"].get(item_key)
        if existing and existing.get("version") == item_version and not force_rebuild:
            stats["items_skipped"] += 1
            continue

        # Extract PDF text
        pdfs_extracted += 1
        if progress_win:
            progress_win.update(
                f"Extracting PDF: {idx + 1}/{len(parent_items)} items ({pdfs_with_text} with text)",
                progress_pct,
                f"{title}..."
            )
        pages = extract_pdf_text_with_pages(item_key)
        if not pages:
            # Index metadata only (title + abstract)
            abstract = item_data.get("abstractNote", "")
            if not title and not abstract:
                stats["items_skipped"] += 1
                continue

            chunks = [{"text": f"{title}. {title}. {abstract}".strip(),
                      "page": 0, "char_start": 0}]
        else:
            # Chunk the PDF text
            pdfs_with_text += 1
            abstract = item_data.get("abstractNote", "")
            chunks = chunk_text_with_pages(pages, title, abstract)

        if not chunks:
            stats["items_skipped"] += 1
            continue

        # Create embeddings for all chunks
        try:
            chunk_texts = [c["text"] for c in chunks]
            embeddings = model.encode(chunk_texts, show_progress_bar=False)

            # Store in index
            index["items"][item_key] = {
                "title": title,
                "version": item_version,
                "indexed_at": datetime.now().isoformat(),
                "chunks": [
                    {
                        "text": chunks[i]["text"],
                        "embedding": embeddings[i].tolist(),
                        "page": chunks[i]["page"],
                        "char_start": chunks[i]["char_start"]
                    }
                    for i in range(len(chunks))
                ]
            }

            stats["items_processed"] += 1
            stats["chunks_created"] += len(chunks)

            if stats["items_processed"] % 5 == 0:
                elapsed = time.time() - start_time
                msg = f"PROGRESS: {stats['items_processed']} items indexed, {stats['chunks_created']} chunks, {elapsed:.0f}s elapsed"
                logger.info(msg)
                stats["status_messages"].append(msg)

        except Exception as e:
            stats["items_failed"] += 1
            stats["errors"].append(f"{item_key}: {str(e)}")
            logger.error(f"Failed to embed {item_key}: {e}")

    # Remove deleted items
    deleted_keys = set(index["items"].keys()) - current_keys
    for key in deleted_keys:
        del index["items"][key]
    stats["items_removed"] = len(deleted_keys)

    # Update metadata and save
    index["created"] = datetime.now().isoformat()
    save_semantic_index(index)

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 2)
    stats["total_items_in_index"] = len(index["items"])
    stats["total_chunks_in_index"] = sum(
        len(item["chunks"]) for item in index["items"].values()
    )
    stats["pdfs_attempted"] = pdfs_extracted
    stats["pdfs_with_text"] = pdfs_with_text

    final_msg = f"COMPLETED: Indexed {stats['items_processed']} items ({pdfs_with_text} PDFs) with {stats['total_chunks_in_index']} chunks in {elapsed:.0f}s"
    stats["status_messages"].append(final_msg)
    logger.info(final_msg)

    # Close progress window
    if progress_win:
        progress_win.update("Complete!", 100)
        import time as time_mod
        time_mod.sleep(1)  # Show completion for 1 second
        progress_win.close()

    return json.dumps(stats, indent=2)


@mcp.tool()
def semantic_search(query: str, limit: int = 10, threshold: float = 0.3) -> str:
    """
    Search the Zotero library using semantic similarity.

    Returns the best matching text chunk from each paper, showing WHY it matched.
    Requires the semantic index to be built first via build_semantic_index().

    Args:
        query: Natural language search query
        limit: Maximum number of results to return (default: 10)
        threshold: Minimum similarity score 0-1 (default: 0.3)

    Returns:
        JSON with matching items, scores, and relevant text chunks
    """
    index = load_semantic_index()

    if not index["items"]:
        return json.dumps({
            "error": "Semantic index is empty",
            "note": "Run build_semantic_index() first to index your library"
        }, indent=2)

    # Embed the query
    model = get_embedding_model()
    query_embedding = model.encode(query)

    # Search all chunks
    results = []
    for item_key, item_data in index["items"].items():
        best_chunk = None
        best_score = -1

        for chunk in item_data["chunks"]:
            # Compute cosine similarity
            chunk_embedding = np.array(chunk["embedding"])
            score = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )

            if score > best_score:
                best_score = score
                best_chunk = chunk

        if best_score >= threshold and best_chunk:
            results.append({
                "key": item_key,
                "title": item_data["title"],
                "score": round(float(best_score), 4),
                "matched_chunk": best_chunk["text"][:300] + ("..." if len(best_chunk["text"]) > 300 else ""),
                "chunk_page": best_chunk["page"]
            })

    # Sort by score and limit
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:limit]

    return json.dumps({
        "query": query,
        "results": results
    }, indent=2)


@mcp.tool()
def get_index_status() -> str:
    """
    Get status information about the semantic search index.

    Returns:
        JSON with index statistics (item count, chunk count, last build time, etc.)
    """
    index = load_semantic_index()

    item_count = len(index.get("items", {}))
    chunk_count = sum(len(item["chunks"]) for item in index.get("items", {}).values())

    # Get file size
    file_size = None
    if INDEX_FILE.exists():
        file_size = INDEX_FILE.stat().st_size

    status = {
        "index_exists": INDEX_FILE.exists(),
        "index_path": str(INDEX_FILE),
        "model": index.get("model"),
        "chunk_size": index.get("chunk_size"),
        "chunk_overlap": index.get("chunk_overlap"),
        "created": index.get("created"),
        "item_count": item_count,
        "chunk_count": chunk_count,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2) if file_size else None
    }

    return json.dumps(status, indent=2)


def main():
    """Main entry point for the server."""
    logger.info("Starting Zotero MCP Server")

    # Initialize Zotero client
    init_zotero_client()

    # Run the MCP server (stdio transport by default)
    mcp.run()


if __name__ == "__main__":
    main()
