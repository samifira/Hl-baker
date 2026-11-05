# SPDX-License-Identifier: GPL-3.0-or-later
"""HL Baker entry point.

  hlbake                -> desktop app
  hlbake bake ...       -> command line bake
  hlbake scan ...       -> show how files would be grouped
  hlbake worker ...     -> internal: used by the desktop app
"""
import sys
import multiprocessing


def main():
    multiprocessing.freeze_support()
    argv = sys.argv[1:]
    if argv and argv[0] == "worker":
        from hlbake.cli import run_worker
        sys.exit(run_worker(argv[1:]))
    if argv and argv[0] in ("bake", "scan", "-h", "--help"):
        from hlbake.cli import main_cli
        sys.exit(main_cli(argv))
    from hlbake.gui import main as gui_main
    sys.exit(gui_main(argv))


if __name__ == "__main__":
    main()
