# Dataset license and attribution

The bundled `optdigits.tra` and `optdigits.tes` are unmodified numeric files from:

Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C50P49.

Source page: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
Downloaded November 8, 2026 from the download link on that page:
https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip

UCI's live dataset page states Creative Commons Attribution 4.0 International:
https://creativecommons.org/licenses/by/4.0/legalcode

The source files are shared under that license, not the project's MIT code
license. Original file and archive SHA-256 values are in `manifest.json`.
Rendered grayscale digits, erased inputs and reconstructed images in the audit are
adaptations of this dataset. Credit remains with the dataset's authors. UCI and
the authors do not endorse this project, its models or its conclusions.

The data contain 64 numeric image-block counts and one numeric class label per
row, not private customer or employer data. Source documentation reports 43
writers: 30 contributed to training and a different 13 to test. Individual writer
IDs are not supplied in these files, so this project cannot independently audit
writer identities or form a writer-disjoint validation split within training.
