# Paper Source

`main.tex` is the arXiv-style source for the AlphaLife-MAS++ paper. The source
uses ACM's `acmart` class in `nonacm` mode and references `references.bib` and
`figures/figure1.png`.

To compile locally:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The compiled PDF is ignored by git so the repository stays source-oriented.
