"""Entry point of the packaged app (Meld.exe): the window, or `--selftest` to check the install."""
import os
import sys
import tempfile
import traceback


def main() -> int:
    if "--selftest" in sys.argv:
        # A windowed app has no console, so the report goes to a file.
        out = sys.argv[sys.argv.index("--selftest") + 1] if len(sys.argv) > sys.argv.index("--selftest") + 1 else None
        path = out or os.path.join(tempfile.gettempdir(), "meld-selftest.txt")
        with open(path, "w", encoding="utf-8") as f:
            def report(line: str) -> None:
                f.write(line + "\n")
                f.flush()

            try:
                from meld import selftest

                return 0 if selftest.run(report, online="--online" in sys.argv) else 1
            except BaseException:  # noqa: BLE001
                report("CRASH\n" + traceback.format_exc())
                return 2

    from meld.gui import main as gui_main

    gui_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
