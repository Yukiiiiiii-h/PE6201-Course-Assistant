# Data explainer

## Source and scope

`CONTENT.zip` is the course-material archive supplied by the project owner. The extracted corpus contains Classes 1-6, slides, pre-reads/summaries, cases, notebooks, coding resources, posters, and guest-lecture material. The current pipeline indexes PDFs only so every result can carry a physical page number.

## Inventory

| Item | Count |
|---|---:|
| Usable files (excluding macOS metadata) | 106 |
| PDF | 85 |
| PDF pages | 1,558 |
| Pages with extracted text | 1,553 |
| Empty/image-only PDF pages | 5 |
| Notebook (`.ipynb`) | 12 |
| Python | 3 |
| Markdown | 1 |
| PNG/JPEG | 5 |

The generated `data/processed/manifest.json` records every PDF's relative path, exact filename, SHA-256, physical page count, extracted-page count, and extraction errors.

## Data flow

1. The original archive was kept unchanged during development but is not duplicated in the instructor ZIP.
2. Its extracted files are included under `data/materials/CONTENT/`.
3. Each PDF is hashed and read page by page.
4. Text is normalized, chunked within its page, and stored with exact file/path/page metadata.
5. SQLite FTS5 builds a local lexical index.

The work package includes the extracted `CONTENT` folder, `manifest.json`, and prebuilt SQLite index so the submitted system can be used immediately. The original ZIP is not duplicated. The instructor confirmed that these project files may be submitted in either a public or private repository.

## Exclusions and responsible use

- `__MACOSX` and `.DS_Store` are ignored as packaging metadata.
- Images, notebook JSON, and Python files are not indexed because the required source locator is a PDF page.
- No student personal data was observed in the intended corpus scope.
- Instructor permission was confirmed for this project, including public or private repository submission. Third-party materials still retain their original ownership and licence status.
