"""
Model Zoo - share and download community models.

Download pre-trained models from GitHub releases or upload your own.
Standardized checkpoint format with metadata for interoperability.

Usage:
    from orca.zoo import Zoo

    # List available models
    Zoo.list()

    # Download a model
    bot = Zoo.download('orca-v3')

    # Upload your model
    Zoo.upload('my_checkpoint.pt', name='phoenix-v1', description='Colony specialist')

    # Package a checkpoint with metadata
    Zoo.package('hex_checkpoint_65.pt', 'my_model.pt',
                name='my-bot', author='Phoenix', elo=1200)
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# Default model registry (GitHub releases)
DEFAULT_REGISTRY_URL = "https://api.github.com/repos/Saiki77/hexbot-building-framework/releases"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


class Zoo:
    """Community model zoo for sharing and downloading hex bots."""

    @staticmethod
    def list(registry_url: str = None, verbose: bool = True) -> List[Dict]:
        """List available models from the registry.

        Returns list of model metadata dicts.
        """
        # List local models
        local = Zoo._list_local()

        # List remote models (from GitHub releases)
        remote = []
        if registry_url or DEFAULT_REGISTRY_URL:
            try:
                remote = Zoo._list_remote(registry_url or DEFAULT_REGISTRY_URL)
            except Exception:
                pass

        all_models = local + remote
        if verbose:
            if not all_models:
                print("No models found. Train one with: python -m orca.train")
                return []
            print(f"{'Name':<25} {'Params':>10} {'ELO':>6} {'Source':<10}")
            print("-" * 55)
            for m in all_models:
                print(f"{m.get('name', '?'):<25} "
                      f"{m.get('params', '?'):>10} "
                      f"{m.get('elo', '?'):>6} "
                      f"{m.get('source', '?'):<10}")
        return all_models

    @staticmethod
    def _list_local() -> List[Dict]:
        """List locally available models."""
        models = []
        if os.path.exists(MODEL_DIR):
            for f in os.listdir(MODEL_DIR):
                if f.endswith('.pt'):
                    meta_path = os.path.join(MODEL_DIR, f.replace('.pt', '.json'))
                    meta = {}
                    if os.path.exists(meta_path):
                        with open(meta_path) as mf:
                            meta = json.load(mf)
                    models.append({
                        'name': meta.get('name', f.replace('.pt', '')),
                        'path': os.path.join(MODEL_DIR, f),
                        'params': meta.get('params', '?'),
                        'elo': meta.get('elo', '?'),
                        'source': 'local',
                        **meta,
                    })

        # Also check standard locations
        for path in ['orca/checkpoint.pt', 'pretrained.pt']:
            if os.path.exists(path):
                models.append({
                    'name': 'orca-default' if 'orca' in path else 'pretrained',
                    'path': path,
                    'params': '3.9M',
                    'elo': '?',
                    'source': 'local',
                })
        return models

    @staticmethod
    def _list_remote(url: str) -> List[Dict]:
        """List models from GitHub releases."""
        try:
            import requests
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            releases = resp.json()
        except Exception:
            return []

        models = []
        for release in releases[:10]:
            for asset in release.get('assets', []):
                if asset['name'].endswith('.pt'):
                    models.append({
                        'name': asset['name'].replace('.pt', ''),
                        'url': asset['browser_download_url'],
                        'size': asset['size'],
                        'params': '?',
                        'elo': '?',
                        'source': 'github',
                        'release': release['tag_name'],
                    })
        return models

    @staticmethod
    def download(name: str, save_dir: str = None) -> str:
        """Download a model by name. Returns path to the downloaded checkpoint.

        Example:
            path = Zoo.download('orca-v3')
            bot = Bot.load(path)
        """
        save_dir = save_dir or MODEL_DIR
        os.makedirs(save_dir, exist_ok=True)

        # Search remote
        try:
            remote = Zoo._list_remote(DEFAULT_REGISTRY_URL)
        except Exception:
            remote = []

        match = None
        for m in remote:
            if name in m.get('name', ''):
                match = m
                break

        if not match:
            raise FileNotFoundError(
                f"Model '{name}' not found. Run Zoo.list() to see available models.")

        url = match['url']
        save_path = os.path.join(save_dir, f"{name}.pt")

        print(f"Downloading {name} from {url}...")
        try:
            import requests
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved to {save_path}")
        except ImportError:
            raise ImportError("Install requests: pip install requests")

        return save_path

    @staticmethod
    def package(checkpoint_path: str, output_path: str,
                name: str = 'my-bot', author: str = 'anonymous',
                elo: int = 0, description: str = '',
                net_config: str = 'standard') -> str:
        """Package a checkpoint with metadata for sharing.

        Creates a .pt file with standardized metadata that the zoo can read.

        Example:
            Zoo.package('hex_checkpoint_65.pt', 'my_model.pt',
                        name='phoenix-v1', author='Phoenix', elo=1200,
                        description='Colony play specialist')
        """
        import torch

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)

        params = sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))

        packaged = {
            'model_state_dict': sd,
            'zoo_metadata': {
                'name': name,
                'author': author,
                'elo': elo,
                'description': description,
                'net_config': net_config,
                'params': params,
                'packaged_at': time.time(),
                'hexbot_version': '4.0.0',
            },
        }
        # Preserve optimizer if present
        if 'optimizer_state_dict' in ckpt:
            packaged['optimizer_state_dict'] = ckpt['optimizer_state_dict']

        torch.save(packaged, output_path)

        # Save metadata JSON alongside
        meta_path = output_path.replace('.pt', '.json')
        with open(meta_path, 'w') as f:
            json.dump(packaged['zoo_metadata'], f, indent=2)

        print(f"Packaged: {output_path} ({params:,} params)")
        print(f"Metadata: {meta_path}")
        return output_path

    @staticmethod
    def load(name_or_path: str, sims: int = 200):
        """Load a model from the zoo as a Bot.

        Accepts a model name (searches zoo) or a file path.

        Example:
            bot = Zoo.load('orca-v3', sims=400)
            move = bot.best_move(game)
        """
        from hexbot import Bot

        if os.path.exists(name_or_path):
            return Bot.load(name_or_path)

        # Search local models
        for m in Zoo._list_local():
            if name_or_path in m.get('name', ''):
                bot = Bot.load(m['path'])
                bot._sims = sims
                return bot

        # Try download
        path = Zoo.download(name_or_path)
        bot = Bot.load(path)
        bot._sims = sims
        return bot
