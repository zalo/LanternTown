#!/usr/bin/env python3
"""Assemble the `parametric-environment` skill from this repository and install it for Claude Code.

    python3 skill/install.py                      # -> ~/.claude/skills/parametric-environment
    python3 skill/install.py --to /some/dir       # anywhere else
    python3 skill/install.py --dry                # list what would be copied

The installed skill is self-contained:

    SKILL.md            the process (read first)
    references/         design docs, rubrics, lessons (from skill/references/ + the repository's docs/)
    templates/          asset / material skeletons and the briefs given to author, judge and specialist agents
    project/            a ready-to-copy project: the toolkit (tk/), the CLI and pipeline (tools/), the three.js
                        runtime, review-render page and gallery (web/), the material library (materials/), every
                        asset generator of the reference run (assets/), the two layout generators (town/, town2/),
                        package.json and the GitHub Pages workflow. Generated output, private reference images,
                        reviews and caches are not copied.

Re-run after changing anything here; the target is replaced whole.
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "skill")
DEFAULT = os.path.expanduser("~/.claude/skills/parametric-environment")

# (source relative to the repository root, destination relative to <target>/project/)
PROJECT = [
    ("tk", "tk"),
    ("tools", "tools"),
    ("materials", "materials"),
    ("assets", "assets"),
    ("town", "town"),
    ("town2", "town2"),
    ("web", "web"),
    ("docs", "docs"),
    (".github", ".github"),
    ("package.json", "package.json"),
    ("package-lock.json", "package-lock.json"),
    ("README.md", "README.md"),
    ("LICENSE", "LICENSE"),
    ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
    ("prompt.md", "prompt.md"),
    (".gitignore", ".gitignore"),
]
# directories / files never copied (name match at any depth) and path prefixes skipped under their source
IGNORE_NAMES = {"__pycache__", ".DS_Store", "node_modules", ".base", ".scratch"}
IGNORE_PREFIX = {
    "docs": ["ref", "img"],                 # private reference boards and the README's pictures
    "web": ["next", "perf", "probetest", "ktx", "ktx/.base", "next/.base", "perf/.base", "glbtown/raw"],   # superseded experiments, local screenshots
    "tools": ["next", "__pycache__"],       # the previous bake generation
    "assets": [],
}
# also skipped: images under assets/ (the reference boards) - only sources travel
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def ignore_for(src_root, rel_prefix):
    skip = set(IGNORE_PREFIX.get(rel_prefix, []))

    def ignore(d, names):
        rel = os.path.relpath(d, src_root)
        rel = "" if rel == "." else rel
        out = set()
        for n in names:
            p = os.path.join(rel, n) if rel else n
            if n in IGNORE_NAMES or p in skip:
                out.add(n)
            elif rel_prefix == "assets" and os.path.splitext(n)[1].lower() in IMAGE_EXT:
                out.add(n)
        return out
    return ignore


def main():
    args = sys.argv[1:]
    target = DEFAULT
    if "--to" in args:
        target = os.path.abspath(args[args.index("--to") + 1])
    dry = "--dry" in args
    plan = [(os.path.join(HERE, "SKILL.md"), os.path.join(target, "SKILL.md")),
            (os.path.join(HERE, "references"), os.path.join(target, "references")),
            (os.path.join(HERE, "templates"), os.path.join(target, "templates"))]
    for src, dst in PROJECT:
        plan.append((os.path.join(ROOT, src), os.path.join(target, "project", dst)))
    # the repository's own docs that the skill's references point at, beside the distilled ones
    for name in ("AUTHORING.md", "QUALITY.md", "TOWN_QUALITY.md", "MATERIALS.md", "LOD.md", "API.md", "API_TEX.md"):
        plan.append((os.path.join(ROOT, "docs", name), os.path.join(target, "references", "docs", name)))
    if dry:
        for s, d in plan:
            print(("DIR " if os.path.isdir(s) else "    ") + os.path.relpath(s, ROOT) + "  ->  " + d)
        return
    if os.path.islink(target) or os.path.isfile(target):
        os.unlink(target)
    elif os.path.isdir(target):
        shutil.rmtree(target)
    os.makedirs(target)
    n = 0
    for s, d in plan:
        if not os.path.exists(s):
            print("missing:", s)
            continue
        if os.path.isdir(s):
            prefix = os.path.relpath(s, ROOT)
            shutil.copytree(s, d, ignore=ignore_for(s, prefix), dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)
        n += 1
    total = 0
    files = 0
    for dp, dn, fn in os.walk(target):
        for f in fn:
            files += 1
            total += os.path.getsize(os.path.join(dp, f))
    print(f"installed {n} entries -> {target}: {files} files, {total / 1e6:.1f} MB")
    print("Claude Code lists it as /parametric-environment after the next session start.")


if __name__ == "__main__":
    main()
