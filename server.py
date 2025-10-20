"""Serve the static documentation bundled with the project."""
from __future__ import annotations

import argparse
import functools
import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Tuple


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start a simple HTTP server that exposes the static documentation."
        )
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Hostname or IP address to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port where the server will listen (default: 8000)",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).parent / "docs",
        help=(
            "Directory to expose over HTTP. The default points to the bundled "
            "documentation."
        ),
    )
    return parser.parse_args()


def validate_directory(directory: Path) -> Path:
    directory = directory.resolve()
    if not directory.is_dir():
        raise SystemExit(f"Directory '{directory}' does not exist or is not a folder")
    index_file = directory / "index.html"
    if not index_file.is_file():
        raise SystemExit(
            f"Directory '{directory}' does not contain an 'index.html' file to serve"
        )
    return directory


def serve_docs(address: Tuple[str, int], directory: Path) -> None:
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(directory))
    with ThreadingHTTPServer(address, handler) as httpd:
        host, port = httpd.server_address
        print(
            "Serving documentation from", directory,
            "on http://{host}:{port}/".format(host=host, port=port),
        )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down the server...")


def main() -> None:
    args = parse_arguments()
    directory = validate_directory(args.directory)
    address = (args.host, args.port)
    try:
        serve_docs(address, directory)
    except OSError as exc:
        if exc.errno == getattr(socket, "EADDRINUSE", None):
            raise SystemExit(
                f"Port {args.port} on host {args.host} is already in use"
            ) from exc
        raise


if __name__ == "__main__":
    main()
