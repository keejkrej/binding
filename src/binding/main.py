from __future__ import annotations

from binding import commands  # noqa: F401
from binding.app import app


def main() -> None:
    app(prog_name="binding")


if __name__ == "__main__":
    main()