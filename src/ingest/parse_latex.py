"""
Parse arXiv LaTeX source packages into structured text.

1. Reads source package and exposes it's .tex files.

2. Latex expansion 

3. cleaning
"""

from __future__ import annotations

import argparse
import gzip
import io
import posixpath
import re
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

LATEX_DIR = ROOT / "data" / "raw" / "latex"

BEGIN_DOCUMENT_RE = re.compile(
    r"\\begin\s*\{\s*document\s*\}",
    re.IGNORECASE,
)

TITLE_RE = re.compile(
    r"\\title\s*(?:\[[^\]]*\])?\s*\{",
    re.IGNORECASE,
)

CITE_RE = re.compile(
    r"\\cite[a-zA-Z]*\s*[\[{]"
)

INCLUDE_RE = re.compile(
    r"\\(?:input|include)\s*\{([^{}]+)\}",
    re.IGNORECASE,
)

CITATION_RE = re.compile(
    r"\\cite[a-zA-Z]*"
    r"(?:\s*\[[^\]]*\])?"
    r"(?:\s*\[[^\]]*\])?"
    r"\s*\{([^{}]+)\}"
)

TEMPLATE_MARKERS = (
    "instructions",
    "formatting",
    "template",
    "bare demo",
    "sample paper",
    "your paper",
    "paper title",
    "guidelines for",
    "author guidelines",
    "style file",
    "latex example",
)


# Return all .tex files contained in an arXiv source package.
def read_tex_files(path: Path) -> dict[str, str]:

    blob = path.read_bytes()

    tex_files: dict[str, str] = {}

    try:

        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:

            for member in tar.getmembers():

                if not member.isfile():

                    continue

                if not member.name.lower().endswith(".tex"):

                    continue

                handle = tar.extractfile(member)

                if handle is None:

                    continue

                text = handle.read().decode("utf-8", errors="replace")

                tex_files[member.name] = text

            return tex_files

    except tarfile.TarError:

        pass

    try:

        text = gzip.decompress(blob).decode("utf-8", errors="replace")

    except OSError as exc:

        raise ValueError(f"Could not read source package: {path}") from exc

    tex_files[path.name] = text

    return tex_files


# Read a brace-balanced group starting at an opening brace.
def read_braced_group(text: str, start: int) -> str:

    depth = 0

    characters: list[str] = []

    for char in text[start:]:

        if char == "{":

            depth += 1

            if depth == 1:

                continue

        elif char == "}":

            depth -= 1

            if depth == 0:

                break

        if depth >= 1:

            characters.append(char)

    return "".join(characters)


# Return the raw LaTeX title of a document, if present.
def extract_title(text: str) -> str | None:

    match = TITLE_RE.search(text)

    if match is None:

        return None

    opening_brace = match.end() - 1

    title = read_braced_group(text, opening_brace)

    title = " ".join(title.split())

    return title or None


# Return True if a title appears to belong to a LaTeX template.
def looks_like_template(title: str) -> bool:

    normalized = title.lower()

    return any(marker in normalized for marker in TEMPLATE_MARKERS)


def choose_main_file(tex_files: dict[str, str]) -> str:
    """Return the most plausible main LaTeX file.

    Candidate detection evaluates each tex file on a comment-stripped view
    so commented-out template titles (``% \\title{emnlp 2017 instructions}``)
    and commented-out ``\\begin{document}`` / ``\\cite`` markers do not
    influence ranking or rejection. The selected file name still flows
    through the downstream pipeline on the original ``tex_files`` content;
    only the candidate-detection view is comment-stripped. ``remove_comments``
    is reused as-is (escaped ``\\%`` is preserved).
    """

    candidates: list[tuple[int, int, str]] = []

    for name, text in tex_files.items():

        candidate_text = remove_comments(text)

        if BEGIN_DOCUMENT_RE.search(candidate_text) is None:

            continue

        title = extract_title(candidate_text)

        if title is not None and looks_like_template(title):

            continue

        citation_count = len(CITE_RE.findall(candidate_text))

        has_title = int(title is not None)

        candidates.append(
            (
                citation_count,
                has_title,
                name,
            )
        )

    if not candidates:

        raise ValueError("Could not identify a main LaTeX file.")

    candidates.sort(reverse=True)

    _, _, main_file = candidates[0]

    return main_file


def resolve_include_path(
    current_file: str,
    target: str,
    tex_files: dict[str, str],
) -> str | None:
    """Resolve an input or include target to a source file."""

    target = target.strip()

    if not target.lower().endswith(".tex"):

        target += ".tex"

    current_directory = posixpath.dirname(current_file)

    candidates = [
        posixpath.normpath(
            posixpath.join(
                current_directory,
                target,
            )
        ),
        posixpath.normpath(target),
    ]

    for candidate in candidates:

        if candidate in tex_files:

            return candidate

    return None


def is_commented_out(
    text: str,
    position: int,
) -> bool:
    """Return True if a position occurs after an active LaTeX comment marker."""

    line_start = text.rfind(
        "\n",
        0,
        position,
    ) + 1

    index = line_start

    while index < position:

        if text[index] == "%":

            backslashes = 0

            previous = index - 1

            while (
                previous >= line_start
                and text[previous] == "\\"
            ):

                backslashes += 1

                previous -= 1

            if backslashes % 2 == 0:

                return True

        index += 1

    return False


def remove_comments(text: str) -> str:
    """Remove LaTeX comments while preserving escaped percent signs."""

    cleaned_lines: list[str] = []

    for line in text.splitlines():

        characters: list[str] = []

        index = 0

        while index < len(line):

            char = line[index]

            if char == "%":

                backslashes = 0

                previous = index - 1

                while (
                    previous >= 0
                    and line[previous] == "\\"
                ):

                    backslashes += 1

                    previous -= 1

                if backslashes % 2 == 0:

                    break

            characters.append(char)

            index += 1

        cleaned_lines.append(
            "".join(characters)
        )

    return "\n".join(cleaned_lines)


def preserve_citations(text: str) -> str:
    """Replace LaTeX citation commands with stable citation markers."""

    def replace_citation(
        match: re.Match[str],
    ) -> str:

        keys = [
            key.strip()
            for key in match.group(1).split(",")
            if key.strip()
        ]

        return f"[[CITE:{','.join(keys)}]]"

    return CITATION_RE.sub(
        replace_citation,
        text,
    )


def resolve_includes(
    tex_files: dict[str, str],
    main_file: str,
) -> tuple[str, list[str]]:
    """Recursively expand input and include commands."""

    warnings: list[str] = []

    def expand_file(
        file_name: str,
        stack: tuple[str, ...],
    ) -> str:

        if file_name in stack:

            cycle = " -> ".join(
                (*stack, file_name)
            )

            warnings.append(
                f"Include cycle detected: {cycle}"
            )

            return ""

        text = tex_files[file_name]

        current_stack = (
            *stack,
            file_name,
        )

        def replace_include(
            match: re.Match[str],
        ) -> str:

            if is_commented_out(
                text,
                match.start(),
            ):

                return match.group(0)

            target = match.group(1)

            resolved = resolve_include_path(
                file_name,
                target,
                tex_files,
            )

            if resolved is None:

                warnings.append(
                    f"Could not resolve include '{target}' "
                    f"from '{file_name}'."
                )

                return match.group(0)

            expanded = expand_file(
                resolved,
                current_stack,
            )

            return f"\n{expanded}\n"

        return INCLUDE_RE.sub(
            replace_include,
            text,
        )

    expanded_text = expand_file(
        main_file,
        (),
    )

    return expanded_text, warnings


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Parse an arXiv LaTeX source package."
    )

    parser.add_argument(
        "--arxiv-id",
        required=True,
        help="arXiv identifier, for example 1412.6980",
    )

    args = parser.parse_args()

    path = LATEX_DIR / f"{args.arxiv_id}.tar.gz"

    if not path.exists():

        raise FileNotFoundError(
            f"Source package not found: {path}"
        )

    tex_files = read_tex_files(path)

    main_file = choose_main_file(tex_files)

    expanded_text, warnings = resolve_includes(
        tex_files,
        main_file,
    )

    cleaned_text = remove_comments(
        expanded_text
    )

    citation_text = preserve_citations(
        cleaned_text
    )

    print(f"Source package: {path.name}")

    print(f"TeX files: {len(tex_files)}")

    for name in sorted(tex_files):

        marker = "*" if name == main_file else " "

        print(f"{marker} {name}")

    print(f"Main file: {main_file}")

    print(
        f"Expanded characters: "
        f"{len(expanded_text)}"
    )

    print(
        f"Cleaned characters: "
        f"{len(cleaned_text)}"
    )

    print(
        f"Citation markers: "
        f"{citation_text.count('[[CITE:')}"
    )

    print(
        f"Warnings: "
        f"{len(warnings)}"
    )

    for warning in warnings:

        print(f"  {warning}")


if __name__ == "__main__":

    main()