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

# Set environment variables BEFORE importing sentence_transformers
# These prevent progress bars and warnings that can slow down MCP subprocess
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Import sentence_transformers at module level (slow in MCP subprocess, but only once at startup)
from sentence_transformers import SentenceTransformer

# Configure logging - both stderr and file
LOG_FILE = Path.home() / ".zotero-mcp" / "server.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOG_FILE, encoding='utf-8')
    ]
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
        logger.info(f"Loading model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        logger.info("Model loaded successfully")
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


# Cache for storage path (avoid repeated lookups)
_cached_storage_path: Optional[str] = None


def get_zotero_storage_path() -> str:
    """
    Get the Zotero storage path, supporting both ZotMoov and default locations.

    Returns the storage path where Zotero/ZotMoov stores attachment files.
    Uses caching to avoid repeated lookups.
    """
    global _cached_storage_path
    if _cached_storage_path is not None:
        return _cached_storage_path

    # Try ZotMoov first
    try:
        import re
        prefs_path = Path.home() / "AppData" / "Roaming" / "Zotero" / "Zotero" / "Profiles"

        # Find the profile directory (usually like r64lmnh5.default)
        if prefs_path.exists():
            for profile_dir in prefs_path.iterdir():
                if profile_dir.is_dir():
                    prefs_js = profile_dir / "prefs.js"
                    if prefs_js.exists():
                        with open(prefs_js, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            # Look for ZotMoov destination directory
                            match = re.search(r'extensions\.zotmoov\.dst_dir["\s]*,\s*"([^"]+)"', content)
                            if match:
                                zotmoov_path = match.group(1)
                                # Convert Windows path format
                                zotmoov_path = zotmoov_path.replace('\\\\', '/').replace('\\', '/')
                                if Path(zotmoov_path).exists():
                                    logger.info(f"Using ZotMoov storage: {zotmoov_path}")
                                    _cached_storage_path = zotmoov_path
                                    return zotmoov_path
    except Exception as e:
        logger.debug(f"Could not get ZotMoov path: {e}")

    # Fall back to default Zotero storage location
    default_path = Path.home() / "Zotero" / "storage"
    if default_path.exists():
        logger.info(f"Using default Zotero storage: {default_path}")
        _cached_storage_path = str(default_path)
        return _cached_storage_path

    logger.warning("Could not find Zotero storage directory")
    return None


# Cache for PDF file list (to avoid repeated network scans)
_pdf_file_cache: Optional[dict] = None  # {normalized_name: full_path}


def _get_pdf_cache(storage_path: str) -> dict:
    """
    Build/return a cache of all PDFs in the storage folder.
    Keys are normalized filenames (lowercase, no extension) for matching.
    """
    global _pdf_file_cache
    if _pdf_file_cache is not None:
        return _pdf_file_cache

    import glob
    logger.info(f"Building PDF file cache from {storage_path}...")

    _pdf_file_cache = {}

    # Try flat structure first (ZotMoov default)
    pdf_files = glob.glob(str(Path(storage_path) / "*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDFs in root folder")

    # Also try recursive search (in case of subfolders)
    if not pdf_files:
        logger.info("No PDFs in root, trying recursive search...")
        pdf_files = glob.glob(str(Path(storage_path) / "**" / "*.pdf"), recursive=True)
        logger.info(f"Found {len(pdf_files)} PDFs recursively")

    for pdf_path in pdf_files:
        filename = Path(pdf_path).stem.lower()  # filename without extension, lowercase
        _pdf_file_cache[filename] = pdf_path

    logger.info(f"PDF cache built: {len(_pdf_file_cache)} files indexed")

    # Log first 3 files for debugging
    if _pdf_file_cache:
        sample = list(_pdf_file_cache.keys())[:3]
        logger.info(f"Sample filenames: {sample}")

    return _pdf_file_cache


def _normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, remove special chars."""
    import re
    # Lowercase and remove special characters except spaces
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def find_pdf_path_for_item(item_key: str, storage_path: str, title: str = "") -> Optional[str]:
    """
    Find the PDF path for an item.
    Returns the path if found, None otherwise.

    Supports:
    - Default Zotero structure: storage_path/ITEM_KEY/filename.pdf
    - ZotMoov with any renaming pattern: matches by title in filename
    """
    import glob
    import os

    # Debug: log what we received
    logger.info(f"find_pdf_path_for_item called: key={item_key}, title_len={len(title) if title else 0}, title={title[:30] if title else 'EMPTY'}...")

    if not storage_path:
        logger.debug(f"No storage path provided for {item_key}")
        return None

    # Normalize path for Windows
    normalized_storage = os.path.normpath(storage_path)

    # Try item_key subfolder first (default Zotero structure)
    subfolder_pattern = os.path.join(normalized_storage, item_key, "*.pdf")
    pdf_files = glob.glob(subfolder_pattern)
    if pdf_files:
        logger.info(f"Found PDF in subfolder for {item_key}")
        return pdf_files[0]

    # Subfolder check failed, try title matching
    logger.info(f"No subfolder match for {item_key}, trying title match. title='{title[:30] if title else 'EMPTY'}'")

    # Try ZotMoov: match by title in filename
    if title:
        logger.info(f"Entering title matching block for {item_key}")
        pdf_cache = _get_pdf_cache(normalized_storage)
        normalized_title = _normalize_for_matching(title)

        # Try to find a PDF whose filename contains a significant part of the title
        # Use first 40 chars of title (truncated titles are common in filenames)
        title_words = normalized_title.split()

        best_match = None
        best_score = 0

        for cached_name, pdf_path in pdf_cache.items():
            normalized_cached = _normalize_for_matching(cached_name)

            # Count how many title words appear in the filename
            matching_words = sum(1 for word in title_words if len(word) > 3 and word in normalized_cached)

            # Require at least 3 matching words (to avoid false positives)
            if matching_words >= 3 and matching_words > best_score:
                best_score = matching_words
                best_match = pdf_path

        if best_match:
            return best_match

    return None


def download_pdfs_to_temp(items_to_index: list, storage_path: str, progress_callback=None) -> tuple[Path, dict]:
    """
    Download PDFs for items that need indexing to a local temp folder.

    Args:
        items_to_index: List of (item_key, title) tuples for items needing indexing
        storage_path: The Zotero/ZotMoov storage path
        progress_callback: Optional callback(current, total, title) for progress updates

    Returns:
        Tuple of (temp_folder_path, {item_key: local_pdf_path} mapping)
    """
    import shutil
    import tempfile

    # Create temp folder
    temp_dir = Path(tempfile.gettempdir()) / "zotero-pdf-cache"
    temp_dir.mkdir(parents=True, exist_ok=True)

    pdf_mapping = {}
    total = len(items_to_index)
    found_count = 0
    not_found_count = 0

    logger.info(f"Starting download of {total} items from storage: {storage_path}")

    # Debug: show first few items to verify titles are present
    if items_to_index:
        for i, item in enumerate(items_to_index[:3]):
            logger.info(f"  Sample item {i}: key={item[0]}, title='{item[1]}'")

    for idx, (item_key, title) in enumerate(items_to_index):
        if progress_callback:
            progress_callback(idx + 1, total, f"Downloading: {title[:40]}")

        # Find PDF on network (match by title since ZotMoov renames files)
        source_pdf = find_pdf_path_for_item(item_key, storage_path, title)
        if not source_pdf:
            not_found_count += 1
            if idx < 5:  # Log first few for debugging
                logger.info(f"No PDF found for {item_key}: {title[:40]}")
            continue
        found_count += 1
        if idx < 5:  # Log first few for debugging
            logger.info(f"Found PDF for {item_key}: {source_pdf}")

        # Copy to temp folder
        dest_pdf = temp_dir / f"{item_key}.pdf"
        try:
            if not dest_pdf.exists():  # Don't re-copy if already there
                shutil.copy2(source_pdf, dest_pdf)
            pdf_mapping[item_key] = str(dest_pdf)
            logger.debug(f"Downloaded {item_key} to temp folder")
        except Exception as e:
            logger.warning(f"Failed to copy PDF for {item_key}: {e}")

    logger.info(f"Download complete: {found_count} PDFs found, {not_found_count} not found, {len(pdf_mapping)} copied to {temp_dir}")
    return temp_dir, pdf_mapping


def extract_pdf_text_from_path(pdf_path: str) -> Optional[list[dict]]:
    """
    Extract text from a PDF file at the given path.
    Returns list of {"page": N, "text": "..."} or None if extraction fails.
    """
    import fitz  # pymupdf

    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append({"page": page_num + 1, "text": text})
        doc.close()
        return pages if pages else None
    except Exception as e:
        logger.debug(f"Failed to extract PDF from {pdf_path}: {e}")
        return None


def extract_pdf_text_with_pages(item_key: str) -> Optional[list[dict]]:
    """
    Extract text from PDF with page tracking (used by get_fulltext_local tool).
    Returns list of {"page": N, "text": "..."} or None if no PDF.
    """
    # Try direct file access first
    storage_path = get_zotero_storage_path()
    if storage_path:
        pdf_path = find_pdf_path_for_item(item_key, storage_path)
        if pdf_path:
            return extract_pdf_text_from_path(pdf_path)

    # Fall back to Zotero local API
    import requests
    try:
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

        if pdf_path:
            return extract_pdf_text_from_path(pdf_path)

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

class SimpleMessageWindow:
    """
    Simple message window that displays a status message.
    Used for showing "Loading model..." before the main progress window.
    """
    def __init__(self, title="Please Wait", message="Loading..."):
        import threading

        self.title = title
        self.message = message
        self.closed = False
        self.root = None

        logger.info(f"{title}: {message}")

        # Start tkinter in separate thread
        self.thread = threading.Thread(target=self._run_window, daemon=True)
        self.thread.start()

        # Give window time to initialize
        import time
        time.sleep(0.3)

    def _run_window(self):
        """Run tkinter window in its own thread."""
        try:
            import tkinter as tk

            self.root = tk.Tk()
            self.root.title(self.title)
            self.root.geometry("400x100")
            self.root.resizable(False, False)

            # Message
            label = tk.Label(self.root, text=self.message, font=("Arial", 12))
            label.pack(expand=True)

            # Keep window on top
            self.root.attributes('-topmost', True)
            self.root.lift()

            # Handle window close
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)

            # Run tkinter event loop
            self.root.mainloop()

        except Exception as e:
            logger.warning(f"Could not create message window: {e}")
            self.root = None

    def _on_close(self):
        """Handle window close button."""
        self.closed = True
        if self.root:
            self.root.destroy()

    def close(self):
        """Close the window."""
        self.closed = True
        if self.root:
            try:
                self.root.after(0, self.root.destroy)
            except:
                pass

        # Give window time to close
        import time
        time.sleep(0.1)


class ProgressWindow:
    """
    Progress window using tkinter that runs in a separate thread.
    Thread-safe updates via queue mechanism.
    """
    def __init__(self, title="Building Index"):
        import threading
        import queue

        self.title = title
        self.queue = queue.Queue()
        self.closed = False
        self.thread = None
        self.root = None

        # Also log to console
        logger.info(f"\n{'='*60}")
        logger.info(f"  {title}")
        logger.info(f"{'='*60}\n")

        # Start tkinter in separate thread
        self.thread = threading.Thread(target=self._run_window, daemon=True)
        self.thread.start()

        # Give window time to initialize
        import time
        time.sleep(0.3)

    def _run_window(self):
        """Run tkinter window in its own thread with event loop."""
        try:
            import tkinter as tk
            from tkinter import ttk

            self.root = tk.Tk()
            self.root.title(self.title)
            self.root.geometry("500x180")
            self.root.resizable(False, False)

            # Title
            title_label = tk.Label(self.root, text=self.title, font=("Arial", 14, "bold"))
            title_label.pack(pady=(15, 10))

            # Status text
            self.status_var = tk.StringVar(value="Initializing...")
            status_label = tk.Label(self.root, textvariable=self.status_var, font=("Arial", 10))
            status_label.pack(pady=5)

            # Progress bar
            self.progress_var = tk.DoubleVar(value=0)
            progress_bar = ttk.Progressbar(
                self.root, variable=self.progress_var, maximum=100, mode='determinate', length=450
            )
            progress_bar.pack(pady=10, padx=25)

            # Details text (item name)
            self.details_var = tk.StringVar(value="")
            details_label = tk.Label(self.root, textvariable=self.details_var, fg="gray", font=("Arial", 9))
            details_label.pack(pady=5)

            # Keep window on top
            self.root.attributes('-topmost', True)
            self.root.lift()

            # Handle window close
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)

            # Start polling queue for updates
            self._poll_queue()

            # Run tkinter event loop
            self.root.mainloop()

        except Exception as e:
            logger.warning(f"Could not create progress window: {e}")
            self.root = None

    def _poll_queue(self):
        """Poll queue for updates (runs in tkinter thread)."""
        if self.closed or self.root is None:
            return

        try:
            while True:
                try:
                    msg = self.queue.get_nowait()
                    if msg.get("close"):
                        self.root.destroy()
                        return
                    if "status" in msg:
                        self.status_var.set(msg["status"])
                    if "progress" in msg:
                        self.progress_var.set(msg["progress"])
                    if "details" in msg:
                        self.details_var.set(msg["details"])
                except:
                    break

            # Schedule next poll (every 50ms)
            self.root.after(50, self._poll_queue)
        except Exception as e:
            logger.debug(f"Poll queue error: {e}")

    def _on_close(self):
        """Handle window close button."""
        self.closed = True
        if self.root:
            self.root.destroy()

    def update(self, current: int, total: int, item_name: str = ""):
        """Send progress update to window (thread-safe)."""
        pct = int((current / total) * 100) if total > 0 else 0

        # Log to console too
        bar_length = 30
        filled = int(bar_length * current // total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)
        msg = f"[{bar}] {current}/{total} ({pct}%)"
        if item_name:
            msg += f" - {item_name[:40]}"
        logger.info(msg)

        # Send to window
        self.queue.put({
            "status": f"Processing {current} of {total} items ({pct}%)",
            "progress": pct,
            "details": item_name[:60] if item_name else ""
        })

    def set_status(self, status: str):
        """Set status message (thread-safe)."""
        logger.info(status)
        self.queue.put({"status": status})

    def close(self):
        """Close the progress window."""
        logger.info(f"\n{'='*60}\n")
        self.closed = True
        self.queue.put({"close": True})

        # Give window time to close
        import time
        time.sleep(0.1)


@mcp.tool()
def build_semantic_index(force_rebuild: bool = False) -> str:
    """
    Build or update the semantic search index from local Zotero library.

    Extracts text from all PDFs, splits into chunks, and creates embeddings.
    PDFs are first downloaded to a local temp folder for faster extraction.

    This operation can take 5-30 minutes depending on library size and PDF count.
    - First run: Downloads ~80MB embedding model
    - PDF download: Copies PDFs from network to local temp folder
    - PDF extraction + embedding: Much faster from local storage

    Args:
        force_rebuild: If True, rebuild entire index. If False, only update changed items.

    Returns:
        JSON with indexing statistics
    """
    import shutil
    import time

    start_time = time.time()
    stats = {"items_processed": 0, "items_skipped": 0, "items_failed": 0,
             "chunks_created": 0, "errors": [], "status_messages": [],
             "pdfs_downloaded": 0, "download_time": 0, "extract_time": 0, "embed_time": 0}

    # Load embedding model FIRST (no GUI - tkinter threads cause ~10x slowdown)
    logger.info("=" * 60)
    logger.info("Loading embedding model (this may take 20-30 seconds on first run)...")
    print("Loading embedding model, please wait...", flush=True)
    model_start = time.time()
    model = get_embedding_model()
    model_load_time = time.time() - model_start
    logger.info(f"Model loaded in {model_load_time:.1f}s")
    print(f"Model loaded in {model_load_time:.1f}s", flush=True)

    # Create progress window (runs in separate thread)
    progress = ProgressWindow("Building Semantic Index")
    progress.set_status("Fetching items from Zotero...")

    try:
        # Use pyzotero to get all items (handles pagination automatically)
        if zot is None:
            init_zotero_client()
        ensure_client()

        progress.set_status("Fetching all items from Zotero...")
        # zot.everything() handles pagination, zot.top() gets parent items only
        all_items = zot.everything(zot.top())
        logger.info(f"Fetched {len(all_items)} items from Zotero")
    except RuntimeError as e:
        progress.close()
        return json.dumps({
            "error": str(e),
            "note": "Check ZOTERO_API_KEY and ZOTERO_USER_ID in .env"
        }, indent=2)
    except Exception as e:
        progress.close()
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

    # Determine which items need indexing
    items_to_index = []
    current_keys = set()
    for item in parent_items:
        item_key = item.get("key")
        item_version = item.get("version", 0)
        title = item.get("data", {}).get("title", "Untitled")[:50]
        current_keys.add(item_key)

        existing = index["items"].get(item_key)
        if existing and existing.get("version") == item_version and not force_rebuild:
            stats["items_skipped"] += 1
        else:
            items_to_index.append((item_key, title, item))

    logger.info(f"Items to index: {len(items_to_index)}, skipped (unchanged): {stats['items_skipped']}")

    if not items_to_index:
        progress.set_status("No items need indexing!")
        progress.close()
        return json.dumps({
            "message": "No items need indexing - all items are up to date",
            "items_skipped": stats["items_skipped"],
            "total_items_in_index": len(index["items"])
        }, indent=2)

    # Get storage path once (cached)
    storage_path = get_zotero_storage_path()

    # Clear PDF cache so we get fresh file list
    global _pdf_file_cache
    _pdf_file_cache = None

    # Phase 1: Download PDFs to local temp folder
    progress.set_status(f"Phase 1: Downloading {len(items_to_index)} PDFs to local temp folder...")
    download_start = time.time()

    def download_progress(current, total, title):
        progress.update(current, total, title)

    temp_dir, pdf_mapping = download_pdfs_to_temp(
        [(key, title) for key, title, _ in items_to_index],
        storage_path,
        download_progress
    )

    stats["download_time"] = round(time.time() - download_start, 2)
    stats["pdfs_downloaded"] = len(pdf_mapping)
    logger.info(f"Downloaded {len(pdf_mapping)} PDFs in {stats['download_time']:.1f}s")

    # Phase 2: Extract text and create embeddings (from local copies)
    progress.set_status(f"Phase 2: Extracting text and creating embeddings...")
    extract_start = time.time()
    pdfs_with_text = 0
    total_embed_time = 0

    for idx, (item_key, title, item) in enumerate(items_to_index):
        item_data = item.get("data", {})
        item_version = item.get("version", 0)

        progress.update(idx + 1, len(items_to_index), f"Processing: {title[:40]}")

        # Extract PDF text from local copy (or try network as fallback)
        pdf_start = time.time()
        local_pdf = pdf_mapping.get(item_key)
        if local_pdf:
            pages = extract_pdf_text_from_path(local_pdf)
        else:
            # No local copy - item might not have a PDF
            pages = None
        pdf_time = time.time() - pdf_start

        if not pages:
            # Index metadata only (title + abstract)
            abstract = item_data.get("abstractNote", "")
            if not title and not abstract:
                continue

            chunks = [{"text": f"{title}. {title}. {abstract}".strip(),
                      "page": 0, "char_start": 0}]
        else:
            # Chunk the PDF text
            pdfs_with_text += 1
            abstract = item_data.get("abstractNote", "")
            chunks = chunk_text_with_pages(pages, title, abstract)

        if not chunks:
            continue

        # Create embeddings for all chunks
        try:
            chunk_texts = [c["text"] for c in chunks]

            embed_start = time.time()
            embeddings = model.encode(chunk_texts, show_progress_bar=False)
            embed_time = time.time() - embed_start
            total_embed_time += embed_time

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

            # Log detailed timing
            has_pdf = pages is not None
            num_pages = len(pages) if pages else 0
            logger.info(f"TIMING [{item_key}]: extract={pdf_time:.2f}s, embed={embed_time:.2f}s, "
                       f"chunks={len(chunks)}, pages={num_pages}, has_pdf={has_pdf}, title={title[:30]}")

        except Exception as e:
            stats["items_failed"] += 1
            stats["errors"].append(f"{item_key}: {str(e)}")
            logger.error(f"Failed to embed {item_key}: {e}")

    stats["extract_time"] = round(time.time() - extract_start - total_embed_time, 2)
    stats["embed_time"] = round(total_embed_time, 2)

    # Phase 3: Cleanup temp folder
    progress.set_status("Cleaning up temp folder...")
    try:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temp folder: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up temp folder: {e}")

    # Remove deleted items from index
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
    stats["pdfs_with_text"] = pdfs_with_text

    final_msg = (f"COMPLETED: Indexed {stats['items_processed']} items ({pdfs_with_text} PDFs) "
                 f"with {stats['total_chunks_in_index']} chunks in {elapsed:.0f}s "
                 f"(download: {stats['download_time']:.0f}s, extract: {stats['extract_time']:.0f}s, embed: {stats['embed_time']:.0f}s)")
    stats["status_messages"].append(final_msg)
    logger.info(f"\n{final_msg}\n")
    progress.close()

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
