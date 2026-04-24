# `paper/` — Manuscript source (mirror)

Optional mirror of the LaTeX source (`main.tex`, `bibliography.bib`,
`itor.cls`, …) that is primarily maintained in the
`knapsack-optimization-paper` repository. Kept here so that the public
release is self-contained for reviewers who want to rebuild the PDF.

To synchronise:

```bash
# from the paper repo
make sync-to-code
# or copy explicitly
cp main.tex bibliography.bib ../knapsack-hld/paper/
```
