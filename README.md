# Tree–network distinguishability fails under the Kimura two-parameter model

Anonymous. 1.1.0-candidate. Unrefereed candidate. DOI: 10.5281/zenodo.23165616.

Start with [AI_INDEX.md](AI_INDEX.md) and [the paper](paper/k2p_tree_network.pdf).

The displayed-tree model with uniform root and edge-specific K2P parameters admits an exact transition-biased continuous-time K2,3/tree coincidence. Three seeds coincide on the larger stochastic region. Anchoring classifies the 83 reduced level-three trinets; universal propagation has a written proof, not a formal certificate. Both models sharing one prescribed ratio greater than one is excluded for K2,3.

## Replay

Python 3.13; install [requirements.txt](requirements.txt) in an isolated environment. Run from the package root:

```sh
python code/verify_release.py --out /tmp/k2p-normal-new
python -O code/verify_release.py --out /tmp/k2p-optimized-new
```

Choose output directories that do not exist. Each run checks all 64 probabilities for three seeds, required admissibility, exact rank-three and rank-nine minors, bijective graph-inventory isomorphisms, all 83 classification/deletion routes and four semantic negative controls. It retains logs, exact minors and vertex mappings.

The graph-atlas inventory was freshly generated during publication using the supplied private reviewer audit; the public command checks it against the submission inventory, rather than rerunning the expensive rooted generator. The private audit code is not redistributed or called independent external assurance. The supplied rooted enumeration remains under code/ for inspection.

Exploratory searches and level-four sampling are not release gates; see [STATUS.json](STATUS.json). [REVIEW_RESPONSE.md](REVIEW_RESPONSE.md) maps the revisions. Original sources and referee documents are not included.

## Build manuscript

With Pandoc and LaTeX installed: from paper/, run `pandoc k2p_tree_network.md --standalone --pdf-engine=pdflatex -o k2p_tree_network.pdf`. The publication PDF uses the equivalent generated LaTeX with Table 1 kept together. The shipped source records that layout repair.
