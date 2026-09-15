"""Allow ``python -m pictor_mcp``."""

from .server import main

if __name__ == "__main__":
    raise SystemExit(main())
