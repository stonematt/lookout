"""Build the wind prototype page from the template and the aggregated data.

Writes ``wind-studies.html`` next to this file: the template with
``wind.json`` injected, wrapped in a document skeleton so it opens directly
in a browser.
"""

import pathlib

HERE = pathlib.Path(__file__).parent


def main() -> None:
    """Inject wind.json into the template and write wind-studies.html."""
    body = (HERE / "wind-template.html").read_text()
    data = (HERE / "wind.json").read_text().strip()
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "</head><body>\n" + body.replace("__DATA__", data) + "\n</body></html>\n"
    )
    (HERE / "wind-studies.html").write_text(page)


if __name__ == "__main__":
    main()
