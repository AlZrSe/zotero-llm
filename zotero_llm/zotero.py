from pyzotero import zotero
from typing import List, Dict, Optional
from datetime import date

class ZoteroClient:
    def __init__(self):
        """Initialize the Zotero client for local HTTP server."""
        self.client = self._create_client()

    def _create_client(self) -> zotero.Zotero:
        """Create and return a Zotero client configured for the local Zotero HTTP server."""
        # Use dummy userID and API key, as local server does not require them
        return zotero.Zotero('0', 'user', 'local-api-key', local=True)

    def fetch_all_items(self) -> Optional[List[Dict]]:
        """Fetch all items from the Zotero library."""

        def parse_item(item: Dict) -> Dict:
            """Parse a single Zotero item into a standardized dictionary."""
            return {
                'zotero_key': item['key'],
                'title': item['data'].get('title', ''),
                'abstract': item['data'].get('abstractNote', ''),
                'authors': item['data'].get('creators', []),
                'year': item['data'].get('date', ''),
                'journal': item['data'].get('publicationTitle', ''),
                'doi': item['data'].get('DOI', ''),
                'keywords': [tag['tag'] for tag in item['data'].get('tags', [])],
            }
        try:
            items = self.client.items(limit=None)  # Fetch all items
            docs = [parse_item(item) for item in items if item['data']['itemType'] != 'attachment']

            # Fetch group libraries
            groups = self.client.groups()
            for group in groups:
                try:
                    group_client = zotero.Zotero(group['id'], 'group', 'local-api-key', local=True)
                    group_items = group_client.items(limit=None)
                    group_docs = [parse_item(item) for item in group_items if item['data']['itemType'] != 'attachment']
                    docs.extend(group_docs)
                except Exception as e:
                    print(f"Error fetching group {group['id']}: {e}")

            # Drop duplicates based on doi
            unique_docs = {doc['doi']: doc for doc in docs if doc['doi']}
            
            return list(unique_docs.values())
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

    def get_zotero_update_date(self) -> Optional[date]:
        """Get the last updated date of the Zotero library."""
        try:
            zotero_main = [self.client.items(limit=1, sort='dateModified', order='desc')]
            groups = self.client.groups()
            for group in groups:
                group_client = zotero.Zotero(group['id'], 'group', 'local-api-key', local=True)
                group_item = group_client.items(limit=1, sort='dateModified', order='desc')
                zotero_main.append(group_item)

            return max([item['data'].get('dateModified', None) for item in zotero_main])
        except Exception as e:
            print(f"Error fetching library info: {e}")
            return None