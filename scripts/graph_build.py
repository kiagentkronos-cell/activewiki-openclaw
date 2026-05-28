#!/usr/bin/env python3
"""ActiveWiki: knowledge-graph builder — thin wrapper around vectordb.py.

All graph logic has been integrated into vectordb.py. This script delegates
to vectordb.py via subprocess for backward compatibility.

Usage (legacy — still works):
  graph_build.py build              Full rebuild of the graph
  graph_build.py build --page <s>   Incremental: extract one page only
  graph_build.py stats              Show graph statistics
  graph_build.py validate           Find orphans, duplicates, issues
  graph_build.py communities build  Community detection + LLM summaries
  graph_build.py communities stats  Community statistics
  graph_build.py communities list   List communities
  graph_build.py communities show   Show a single community

Usage (unified — preferred):
  vectordb.py build --graph --communities    Full pipeline: vectors + graph + communities
  vectordb.py graph build                    Graph only
  vectordb.py graph stats                    Graph statistics
  vectordb.py graph validate                 Validate graph integrity
  vectordb.py graph search <query>           Search entities
  vectordb.py graph communities build        Community detection
  vectordb.py graph communities list         List communities
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

VECTORDB = Path(__file__).resolve().parent / "vectordb.py"


def main() -> int:
    if not VECTORDB.exists():
        print(f"[error] vectordb.py not found at {VECTORDB}", file=sys.stderr)
        return 1

    # Map legacy commands to vectordb.py equivalents
    args = sys.argv[1:]
    mapped: list[str] = ["graph"]

    if not args:
        mapped.append("stats")
    elif args[0] == "build":
        mapped.append("build")
        # Pass --page through
        if len(args) > 1:
            for i in range(1, len(args), 2):
                if i < len(args):
                    mapped.append(args[i])
                    if i + 1 < len(args):
                        mapped.append(args[i + 1])
    elif args[0] == "stats":
        mapped.append("stats")
    elif args[0] == "validate":
        mapped.append("validate")
    elif args[0] == "communities" and len(args) > 1:
        mapped.append("communities")
        mapped.append(args[1])
        # Pass remaining args
        for a in args[2:]:
            mapped.append(a)
    else:
        print(f"[error] Unknown command: {' '.join(args)}", file=sys.stderr)
        print(f"       Use: build [--page <slug>] | stats | validate | communities <subcmd>",
              file=sys.stderr)
        return 1

    result = subprocess.run(
        [sys.executable, str(VECTORDB)] + mapped,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
