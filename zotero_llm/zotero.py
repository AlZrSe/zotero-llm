from pyzotero import zotero
from typing import List, Dict, Optional
import os

class ZoteroClient:
    def __init__(self):
        """Initialize the Zotero client for local HTTP server."""
        self.client = self._create_client()

    def _create_client(self) -> zotero.Zotero:
        """Create and return a Zotero client configured for the local Zotero HTTP server using ZOTERO_URL."""
        zotero_url = os.environ.get('ZOTERO_URL')
        if not zotero_url:
            raise ValueError("ZOTERO_URL environment variable not set.")
        # Use dummy userID and API key, as local server does not require them
        zot = zotero.Zotero('0', 'user', 'local-api-key', local=True)
        zot.endpoint = zotero_url
        return zot

    def fetch_all_items(self) -> Optional[List[Dict]]:
        """Fetch all items from the Zotero library."""
        try:
            items = self.client.items(limit=5000)  # Fetch up to 5000 items
            docs = [
                {
                    'zotero_key': item['key'],
                    'title': item['data'].get('title', ''),
                    'abstract': item['data'].get('abstractNote', ''),
                    'authors': item['data'].get('creators', []),
                    'year': item['data'].get('date', ''),
                    'journal': item['data'].get('publicationTitle', ''),
                    'doi': item['data'].get('DOI', ''),
                    'keywords': [tag['tag'] for tag in item['data'].get('tags', [])],
                }
                for item in items if item['data']['itemType'] != 'attachment'
            ]
            
            print(f"Fetched {len(docs)} items from Zotero.")
            return docs
        except Exception as e:
            print(f"Error fetching items: {e}")
            return None

    def test_connection(self) -> bool:
        """Test the connection to Zotero server."""
        try:
            self.client.count_items()
            return True
        except Exception:
            return False