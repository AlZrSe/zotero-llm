from pyzotero import zotero


def get_zotero_client():
    """Return a Zotero client configured for the local Zotero HTTP server."""
    # Use dummy userID and API key, as local server does not require them
    zot = zotero.Zotero('0', 'user', 'local-api-key', local=True)
    # zot.base_url = 'http://127.0.0.1:23119/zotero/api'
    return zot

def fetch_all_items(zot):
    """Fetch all items from the Zotero library."""
    try:
        items = zot.items(limit=5000)  # Fetch up to 5000 items
        docs = [{'zotero_key': item['key'],
                 'title': item['data'].get('title', ''),
                 'abstract': item['data'].get('abstractNote', ''),
                 'authors': item['data'].get('creators', []),
                 'year': item['data'].get('date', ''),
                 'journal': item['data'].get('publicationTitle', ''),
                 'doi': item['data'].get('DOI', ''),
                 'keywords': [tag['tag'] for tag in item['data'].get('tags', [])],
                } for item in items if item['data']['itemType'] != 'attachment']
        
        print(f"Fetched {len(docs)} items from Zotero.")
        return docs
    except Exception as e:
        print(f"Error fetching items: {e}")
        return None