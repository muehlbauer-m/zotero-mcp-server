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
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from dotenv import load_dotenv
from pyzotero import zotero
from mcp.server.fastmcp import FastMCP

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

    # Assign to collection before creation
    if collection_key:
        template["collections"] = [collection_key]

    # Create item
    response = zot.create_items([template])

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

    # Get item for deletion
    item = zot.item(item_key)
    zot.delete_item(item)

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


def main():
    """Main entry point for the server."""
    logger.info("Starting Zotero MCP Server")

    # Initialize Zotero client
    init_zotero_client()

    # Run the MCP server (stdio transport by default)
    mcp.run()


if __name__ == "__main__":
    main()
