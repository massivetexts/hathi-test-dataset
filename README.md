# HathiTrust Testing Dataset

This repo contains a 143k volume sample of the English-language HathiTrust catalogue, to use as a test set. It aims to have individual authors well-represented: rather than randomly sampling volumes, authors were randomly sampled and the volumes were taken from that selection.

- English
- Focus only on monographic books (bib_fmt=='BK') and non-government docs
- Authors with more than 10 works. Inclusion in dataset capped at 100
- a 150k selection is made, then cross-referenced with books available in the [Extracted Features Dataset](https://analytics.hathitrust.org/datasets)
