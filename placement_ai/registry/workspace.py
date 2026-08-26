"""
placement_ai/registry/workspace.py
----------------------------------
Tenancy: one directory per organisation.

    workspaces/
      acme-college-4f2a/
        workspace.json      name, access code hash, timestamps
        models/<version>/   one trained bundle per directory
        history.db          this workspace's predictions
        datasets/           snapshots of what was uploaded

There is deliberately no central index file. Listing scans directories, so two
people creating a workspace at the same moment cannot corrupt a shared registry
— a real risk in Streamlit, where every browser session runs the same script.

On the access code: it separates tenants, it is not authentication. The hash
stops a code being read out of the directory, but anyone with filesystem access
can read every workspace's data. That trade is fine for a single-team
deployment and is documented rather than dressed up; putting this in front of
real students means putting a real identity provider in front of it first.
"""

from __future__ import annotations

import hashlib
import secrets
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from placement_ai.config import WORKSPACE_ROOT
from placement_ai.utils import read_json, slugify, utc_now_iso, write_json

WORKSPACE_FILE = "workspace.json"
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no I/O/0/1 to survive being read aloud
CODE_LENGTH = 8


class WorkspaceError(RuntimeError):
    """Something went wrong opening or creating a workspace."""


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.strip().upper().encode("utf-8")).hexdigest()


def generate_code() -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))


@dataclass
class Workspace:
    id: str
    name: str
    description: str
    created_at: str
    root: Path
    active_model: str | None = None
    # repr=False so the hash never lands in a log line or a Streamlit debug dump.
    _code_hash: str = field(default="", repr=False)

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
                "code_hash": self._code_hash,
            },
        )

    def verify_code(self, code: str) -> bool:
        return secrets.compare_digest(self._code_hash, _hash_code(code))

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
        workspace = Workspace(
            id=str(data.get("id", directory.name)),
            name=str(data.get("name", directory.name)),
            description=str(data.get("description", "")),
            created_at=str(data.get("created_at", "")),
            root=directory,
            active_model=data.get("active_model"),
        )
        workspace._code_hash = str(data.get("code_hash", ""))
        return workspace

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

    def create(
        self,
        name: str,
        description: str = "",
        access_code: str | None = None,
    ) -> tuple[Workspace, str]:
        """Create a workspace, returning it with its access code in the clear.

        The plaintext code is returned exactly once — only its hash is stored,
        so a lost code cannot be recovered, only reset.
        """
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

        code = (access_code or generate_code()).strip().upper()
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
        workspace._code_hash = _hash_code(code)
        workspace.save()
        return workspace, code

    def open(self, workspace_id: str, access_code: str) -> Workspace:
        workspace = self.get(workspace_id)
        if workspace is None:
            raise WorkspaceError("That workspace no longer exists.")
        if not workspace.verify_code(access_code):
            raise WorkspaceError("That access code does not match this workspace.")
        return workspace

    def reset_code(self, workspace_id: str) -> str:
        """Issue a new access code. Whoever can reach the disk can do this."""
        workspace = self.get(workspace_id)
        if workspace is None:
            raise WorkspaceError("That workspace no longer exists.")
        code = generate_code()
        workspace._code_hash = _hash_code(code)
        workspace.save()
        return code

    def delete(self, workspace_id: str, access_code: str) -> None:
        """Remove a workspace and everything in it. There is no undo."""
        workspace = self.open(workspace_id, access_code)
        shutil.rmtree(workspace.root)
