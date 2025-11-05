"""Utility to validate git tags against pyproject.toml version."""

from logging import INFO, Logger, basicConfig, getLogger
from pathlib import Path
from sys import exit as sys_exit

from git import GitCommandError, Repo
from rich.console import Console
from rich.logging import RichHandler
from tomli import loads

basicConfig(
    level=INFO,
    handlers=[
        RichHandler(
            level=INFO,
            console=Console(),
            rich_tracebacks=True,
        ),
    ],
    format="%(message)s",
)
logger: Logger = getLogger(__name__)


def read_version_from_pyproject(path: Path) -> str:
    """Return the package version from pyproject.toml (PEP 621 style for uv)."""
    if not path.exists():
        msg: str = f"{path} does not exist."
        raise FileNotFoundError(msg)
    data: dict = loads(path.read_text(encoding="utf-8"))
    project: dict = data["project"]
    return project["version"]


def get_exact_tag_for_head(repo: Repo) -> str | None:
    """Return exact tag for HEAD commit, or None."""
    try:
        commit_hex: str = repo.head.commit.hexsha
    except Exception:
        return None
    try:
        return repo.git.describe("--tags", "--exact-match", commit_hex)
    except GitCommandError:
        return None


def main() -> int:
    """Validate git tag against pyproject.toml version."""
    repo_path: Path = Path.cwd().parent.parent
    repo: Repo = Repo(repo_path)
    logger.info("Repository at %s", repo.working_tree_dir)
    try:
        branch: str = repo.active_branch.name
    except Exception:
        # detached HEAD or no branch
        logger.info("Detached HEAD or unknown branch.")
        return 0
    if branch != "main":
        logger.info("Not on main branch")
        return 0
    tag: str | None = get_exact_tag_for_head(repo)
    if not tag:
        logger.info("No tag found for the current commit.")
        return 0
    version: str = read_version_from_pyproject(repo_path / "pyproject.toml")
    if not version:
        logger.error("pyproject.toml not found or version not set.")
        return 1
    if tag != version:
        logger.error(
            "Tag '%s' does not match version '%s' in pyproject.toml.",
            tag,
            version,
        )
        return 1
    logger.info("Tag '%s' matches version '%s'.", tag, version)
    return 0


if __name__ == "__main__":
    sys_exit(main())
