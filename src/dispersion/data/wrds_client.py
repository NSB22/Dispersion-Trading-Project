"""
WRDS connection plumbing.

Usage:
    from dispersion.data.wrds_client import get_connection
    db = get_connection()
"""
import os
from dotenv import load_dotenv
import wrds

load_dotenv()


def get_connection() -> wrds.Connection:
    """Open a WRDS connection (username read from .env)."""
    username = os.getenv("WRDS_USERNAME")
    if not username:
        raise RuntimeError("WRDS_USERNAME missing. Add it to the .env file.")
    return wrds.Connection(wrds_username=username)


def test_connection() -> None:
    """Smoke test: check access to crsp, optionm, comp libraries."""
    db = get_connection()
    print("WRDS connection established.")
    libs = db.list_libraries()
    for lib in ["crsp", "optionm", "comp"]:
        print(f"  - {lib}: {'OK' if lib in libs else 'ABSENT'}")
    db.close()


if __name__ == "__main__":
    test_connection()