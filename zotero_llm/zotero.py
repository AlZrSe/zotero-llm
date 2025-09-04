from pyzotero import zotero
from typing import List, Dict, Optional, Iterator
from datetime import date
from tqdm import tqdm

class ZoteroClient:
    def __init__(self, user_id: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Zotero client.
        
        Args:
            user_id: Zotero user ID (required for web API)
            api_key: Zotero API key (required for web API)
        """
        self.user_id = user_id
        self.api_key = api_key

    def get_client(self, library_type: str = 'user', group_id: Optional[str] = None) -> zotero.Zotero:
        """Create and return a Zotero client."""
        if self.user_id:
            if not self.api_key:
                raise ValueError("user_id and api_key are required when using web API")
            if library_type == 'user':
                return zotero.Zotero(self.user_id, library_type, self.api_key)
            elif library_type == 'group' and group_id:
                return zotero.Zotero(group_id, 'group', self.api_key)
            else:
                raise ValueError("group_id is required when library_type is 'group'")
        else:
            # Use dummy userID and API key, as local server does not require them
            if library_type == 'user':
                return zotero.Zotero('0', library_type, 'local-api-key', local=True)
            elif library_type == 'group' and group_id:
                return zotero.Zotero(group_id, 'group', 'local-api-key', local=True)
            else:
                raise ValueError("group_id is required when library_type is 'group'")

    def fetch_items_paginated(self, client: zotero.Zotero, batch_size: int = 100) -> Iterator[List[Dict]]:
        """
        Fetch items from Zotero library with pagination.
        
        Args:
            client: Zotero client
            batch_size: Number of items to fetch per batch
            
        Yields:
            List of parsed items for each batch
        """
        def parse_item(item: Dict) -> Dict:
            """Parse a single Zotero item into a standardized dictionary."""
            return {
                'zotero_key': item['key'],
                'title': item['data'].get('title', ''),
                'abstract': item['data'].get('abstractNote', ''),
                'authors': item['data'].get('creators', []),
                'year': item['data'].get('date', ''),
                'journal': item['data'].get('publicationTitle', ''),
                'doi': item['data'].get('DOI', item['data'].get('ISBN', '')),
                'keywords': [tag['tag'] for tag in item['data'].get('tags', [])],
            }

        try:
            if not self.user_id:
                items = client.items(limit=None)
                batch_docs = [parse_item(item) for item in items if item['data']['itemType'] != 'attachment']
                if batch_docs:
                    yield batch_docs
            else:
            # Get main library items with pagination
                offset = 0
                while True:
                    items = client.items(limit=batch_size, start=offset)
                    if not items:
                        break
                    
                    # Parse and filter items
                    batch_docs = [parse_item(item) for item in items if item['data']['itemType'] != 'attachment']
                    if batch_docs:
                        yield batch_docs
                    
                    # If we got fewer items than batch_size, we've reached the end
                    if len(items) < batch_size:
                        break
                    
                    offset += len(batch_docs)
                    
        except Exception as e:
            print(f"Error in fetch_items_paginated: {e}")
            yield []

    def fetch_all_items(self, batch_size: int = 100, include_groups: bool = True) -> Optional[List[Dict]]:
        """
        Fetch all items from the Zotero library using pagination.
        
        Args:
            batch_size: Number of items to fetch per batch
            include_groups: Whether to include group libraries
            
        Returns:
            List of all parsed items or None if error occurred
        """
        try:
            all_docs = []

            user_client = self.get_client(library_type='user')
            
            # Fetch items in batches
            with tqdm(desc="Fetching Zotero user items", unit=" items") as pbar:
                for batch in self.fetch_items_paginated(user_client, batch_size=batch_size):
                    all_docs.extend(batch)
                    pbar.update(len(batch))

            # Fetch group libraries
            groups = user_client.groups()
            for group in groups:
                try:
                    group_client = self.get_client(library_type='group', group_id=group['id'])
                    with tqdm(desc=f"Fetching Zotero group \"{group['data']['name']}\" items", unit=" items") as pbar:
                        for batch in self.fetch_items_paginated(group_client, batch_size=batch_size):
                            all_docs.extend(batch)
                            pbar.update(len(batch))

                except Exception as e:
                    print(f"Error fetching group {group['id']}: {e}")
            
            # Drop duplicates based on doi
            unique_docs = {doc['doi']: doc for doc in all_docs if doc.get('doi', None)}

            return list(unique_docs.values())
            
        except Exception as e:
            print(f"Error fetching all items: {e}")
            return None

    def fetch_items_with_callback(self, callback, library_type: str = 'user', 
                                 batch_size: int = 100, include_groups: bool = True) -> bool:
        """
        Fetch items and process them with a callback function.
        
        Args:
            callback: Function to call with each batch of items
            library_type: Type of library ('user' or 'group')
            batch_size: Number of items to fetch per batch
            include_groups: Whether to include group libraries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for batch in self.fetch_items_paginated(library_type, batch_size, include_groups):
                if not callback(batch):
                    break  # Stop if callback returns False
            return True
        except Exception as e:
            print(f"Error in fetch_items_with_callback: {e}")
            return False

    def test_connection(self) -> bool:
        """Test the connection to Zotero server."""
        try:
            self.get_client('user').count_items()
            return True
        except Exception:
            return False

    def get_zotero_update_date(self) -> Optional[date]:
        """Get the last updated date of the Zotero library."""
        try:
            main_client = self.get_client('user')
            zotero_main = [main_client.items(limit=1, sort='dateModified', order='desc')]
            
            # Fetch group libraries
            try:                
                groups = main_client.groups()
                for group in groups:
                    try:
                        group_client = self.get_client('group', group['id'])                        
                        group_item = group_client.items(limit=1, sort='dateModified', order='desc')
                        zotero_main.append(group_item)
                    except Exception as e:
                        print(f"Error fetching group {group['id']}: {e}")
            except Exception as e:
                print(f"Error fetching groups: {e}")

            return max([item['data'].get('dateModified', None) for item in zotero_main])
        except Exception as e:
            print(f"Error fetching library info: {e}")
            return None

    def get_library_info(self, library_type: str = 'user') -> Optional[Dict]:
        """Get information about the Zotero library."""
        try:
            client = self.get_client(library_type)
            
            if self.user_id:
                # For web API, we can get user info
                if library_type == 'user':
                    user_info = client.user()
                    return {
                        'type': 'user',
                        'name': user_info.get('name', ''),
                        'username': user_info.get('username', ''),
                        'total_items': client.count_items()
                    }
                else:
                    # For group libraries
                    group_info = client.group()
                    return {
                        'type': 'group',
                        'name': group_info.get('data', {}).get('name', ''),
                        'total_items': client.count_items()
                    }
            else:
                # For local server, return basic info
                return {
                    'type': 'local',
                    'name': 'Local Zotero Server',
                    'total_items': client.count_items()
                }
        except Exception as e:
            print(f"Error fetching library info: {e}")
            return None