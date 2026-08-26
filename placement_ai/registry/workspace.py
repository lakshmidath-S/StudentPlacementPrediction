"""
placement_ai/registry/workspace.py
----------------------------------
Tenancy: one directory per organisation.

    workspaces/
      acme-college-4f2a/
        workspace.json      name, description, timestamps
        models/<version>/   one trained bundle per directory
        history.db          this workspace's predictions
        datasets/           snapshots of what was uploaded

There is deliberately no central index file. Listing scans directories, so two
people creating a workspace at the same moment cannot corrupt a shared registry
— a real risk in Streamlit, where every browser session runs the same script.

There is no access code and no login. A workspace separates one organisation's
data from another's in the UI; it never was authentication — the hash it used to
carry stopped nobody with filesystem access from reading every workspace on
disk, while costing an ordinary user a code to lose. Putting this in front of
real students means putting a real identity provider in front of it first, and
that provider is what should decide who may open which workspace.
"""

from __future__ import annotations

import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path

from placement_ai.config import WORKSPACE_ROOT
from placement_ai.utils import read_json, slugify, utc_now_iso, write_json

WORKSPACE_FILE = "workspace.json"


class WorkspaceError(RuntimeError):
    """Something went wrong opening or creating a workspace."""


@dataclass
class Workspace:
    id: str
    name: str
    description: str
    created_at: str
    root: Path
    active_model: str | None = None

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def history_path(self) -> Path:
        return self.root / "history.db"

    @property
    def config_path(self) -> Path:
        return self.root / WORKSPACE_FILE

    def save(self) -> None:
        write_json(
            self.config_path,
            {
                "id": self.id,
                "name": self.name,
                "description": self.description,
                "created_at": self.created_at,
                "active_model": self.active_model,
            },
        )

    def set_active_model(self, version: str | None) -> None:
        self.active_model = version
        self.save()


class WorkspaceStore:
    """Creates, lists and opens workspaces under one root directory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or WORKSPACE_ROOT)

    def _load(self, directory: Path) -> Workspace | None:
        config = directory / WORKSPACE_FILE
        if not config.exists():
            return None
        try:
            data = read_json(config)
        except (ValueError, OSError):
            return None
        # A workspace written before access codes were dropped still carries a
        # "code_hash" key here. Ignoring it is the whole migration; the next
        # save drops it.
        return Workspace(
            id=str(data.get("id", directory.name)),
            name=str(data.get("name", directory.name)),
            description=str(data.get("description", "")),
            created_at=str(data.get("created_at", "")),
            root=directory,
            active_model=data.get("active_model"),
        )

    def list(self) -> list[Workspace]:
        """Every workspace on disk, newest first."""
        if not self.root.exists():
            return []
        found = [
            workspace
            for directory in sorted(self.root.iterdir())
            if directory.is_dir() and (workspace := self._load(directory)) is not None
        ]
        return sorted(found, key=lambda w: w.created_at, reverse=True)

    def get(self, workspace_id: str) -> Workspace | None:
        return self._load(self.root / workspace_id)

    def create(self, name: str, description: str = "") -> Workspace:
        """Create a workspace directory and the config that makes it one."""
        name = name.strip()
        if not name:
            raise WorkspaceError("A workspace needs a name.")

        slug = slugify(name, "workspace")
        # A short random suffix keeps two workspaces with the same name apart
        # without making the directory unreadable.
        workspace_id = f"{slug[:32]}-{secrets.token_hex(2)}"
        directory = self.root / workspace_id
        if directory.exists():
            raise WorkspaceError(f"A workspace directory already exists at {directory}.")

        directory.mkdir(parents=True, exist_ok=True)
        (directory / "models").mkdir(exist_ok=True)
        (directory / "datasets").mkdir(exist_ok=True)

        workspace = Workspace(
            id=workspace_id,
            name=name,
            description=description.strip(),
            created_at=utc_now_iso(),
            root=directory,
        )
        workspace.save()
        return workspace

    def open(self, workspace_id: str) -> Workspace:
        workspace = self.get(workspace_id)
        if workspace is None:
            raise WorkspaceError("That workspace no longer exists. Create one to continue.")
        return workspace

    def delete(self, workspace_id: str) -> None:
        """Remove a workspace and everything in it. There is no undo."""
        workspace = self.open(workspace_id)
        shutil.rmtree(workspace.root)
