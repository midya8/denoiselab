# DenoiseLab protocol, equations and evidence

## Scope

Original small implementation of established denoising reconstruction and
observed-coordinate controls. It trains a real nonlinear predictor on public
UCI optical digits; it is neither a classifier nor a probabilistic VAE. No
pretraining, image-file ingestion, causal inference or new algorithm.

## Provenance and partitions

Unmodified `optdigits.tra` (3,823 rows) and `optdigits.tes` (1,797) retain their
original SHA-256 values from the UCI archive. Each row contains 64 integer
block counts 0..16 plus a digit label. Authors produced 8-by-8 counts from
32-by-32 bitmaps; this project divides counts by the known constant 16.
This is not 28-by-28 MNIST. The scikit-learn 1,797-row digits pool is not used
as a substitute for the original training and test files.

For each class in original training, hash-sort the 64 original feature bytes,
then hold out the first floor(n/5) rows. Ties use source row index. Sorted
original fitting and validation indices are saved, giving 3,062/761 rows.
Labels are used only for this split, not as predictors, fitting targets,
nearest-neighbor filters or corruption inputs. Exact feature-set intersections
are empty for fit/validation, and for original train/test. That does not rule
out near duplicates or similar writer styles. UCI describes 30 training and 13
other test writers; individual identities are unavailable, so validation
writer separation and original writer identities cannot be independently checked.

Raw data and rendered/erased/reconstructed adaptations are CC BY 4.0, not MIT.
Original code is MIT. The current UCI page's license section was checked on
November 8, 2026. No dataset author endorsement is implied. Data are reused
from DigitLab's verified numeric archive, not downloaded privately or invented.

## Mask semantics and leakage boundary

`mask[j]=1` means pixel j is missing. Predictor input is
`z0=concat(where(mask,0,x), mask)`. Explicit mask flags distinguish a known
zero from a missing pixel. Every predictor discards the hidden entries before
using x, even when a caller supplied the complete target array. The target
is retained separately only for the training loss and evaluation score.

Output for visible coordinates is always copied from input, not changed by
network, PCA, neighbor or mean. The hidden-only error is
`sum_j mask_j*(prediction_j-x_j)^2 / sum_j mask_j` per image; the report averages
this across images. Each evaluation row has an equal hidden count. Thus pooled
and image-mean hidden errors agree here, but that equivalence is not promised
for variable counts. All-pixel error is hidden error times hidden_count/64,
which is explicitly not the primary metric. No-hidden masks return identity
and null score in the browser; training rejects a batch with no missing pixel.
All-hidden inference is allowed, but is an out-of-training-mask extreme.

## Network and derivatives

Dimensions: 128->64->16->64->64. Three tanh activations, then stable sigmoid.
There are 14,544 learned scalar parameters. Initial weights use independent
seeded Gaussian draws with standard deviation sqrt(1/fan_in); biases start zero.

For batch x, mask m, raw prediction p, N=sum(m), and weight-only coefficient
lambda=1e-5:

```
L = sum(m*(p-x)^2)/N + lambda/2 * sum_l ||W_l||_F^2
D_last = (2/N)*m*(p-x)*p*(1-p)
grad_W_l = H_(l-1).T @ D_l + lambda*W_l
grad_b_l = sum_rows D_l
D_(l-1) = (D_l @ W_l.T) * (1-H_(l-1)^2)
```

The mask does not multiply the decoder's visible activations into loss; they
contribute zero by m. The displayed pass-through output is not fed back into
the hidden-only training derivative. Biases are not penalized. No autograd,
gradient clipping, dropout, sigmoid logloss or auxiliary label supervision.

Adam has beta1=.9, beta2=.999, epsilon=1e-8, bias correction by global update
count and learning rate .003. Batch size is 128; final short batch uses its own
actual missing count. Training is 60 epochs, shuffle seed 41 and corruption
seed 51 in separate NumPy generators. Each presentation hides exactly 32
uniformly chosen coordinates without replacement, refreshed each minibatch.
There is no global RNG mutation.

Network initialization seed 23 is fixed. Candidate checkpoints at epochs
10/30/60 use one held-fixed validation mask per image, seed 101, 32 hidden
coordinates. Minimum mean hidden MSE selects a checkpoint; earliest listed
wins ties. Epoch 60 wins; all candidates, sampled last-epoch train objectives
and validation scores are retained. A sampled train objective contains L2 and
is not the same quantity as held-fixed validation hidden-only MSE.

Architecture, seeds, masks and candidate grid are in `protocol.json`. The
`evaluation-plan.json` is written before any test-loader call, and test loading
is prohibited by a patched-loader assertion during complete refits. After
selection, model hashes bind `selection.json`; evaluation checks those and
protocol/plan before loading test examples. Hashes detect accidental changed
artifacts, not a maliciously rewritten implementation.

## Strong controls

All controls fit only the same 3,062 uncorrupted fitting rows.

1. Pixel mean fills missing entries with the fitting coordinate mean.
2. PCA uses centered fitting SVD, ranks 8/16/32, and fixes each basis sign by
   making its largest-magnitude coordinate positive. Ridge candidates are
   .001/.01/.1/1. For each query's observed index set O and basis B (rank by 64),
   solve `(B_O B_O.T + ridge I)a = B_O(x_O - mean_O)`.
   Fill missing entries with `clip(mean + a B,0,1)`, while preserving visible
   entries. This is a regularized observed-coordinate solve, not a projection
   of zero-filled test targets. A separate augmented least-squares oracle
   checks the NumPy solve; the browser independently uses Cholesky. Positive
   ridge makes the coefficient system positive definite even with no observations.
   Validation hidden MSE selects rank/ridge on the same masks as the network;
   lowest listed candidate wins ties. Rank 16, ridge .1 wins. A separately
   selected rank-16 control is saved; here it is identical to the global winner,
   so the display does not pretend it is another independent method.
3. Exact nearest fitting example minimizes squared distance only over visible
   pixels. Lowest fitting index breaks ties. Hidden outputs copy that example.
   It receives no true digit label, hidden target or all-pixel distance. An
   all-hidden query chooses the first fitting example by the documented tie.
   Data coordinates are multiples of 1/16; their squared distances/sums here
   are binary-exact, so the native and scalar browser distance ties agree.

PCA rank and neural code width happen to match, but parameter counts, inductive
biases, training objectives and compute differ. These are descriptive controls,
not an isolated causal test of nonlinearity or a matched-compute benchmark.
The nearest-neighbor control is deliberately not weakened to make a neural win.

## Frozen test plan and complete outcomes

Four masks per original test image, independently generated by their fixed
case seed: random16/201, random32/202, random48/203 and block16/204. Random
masks use a uniform subset of the stated count without replacement. Block
masks choose each top-left coordinate uniformly from 0..4 and hide a 4x4 block.
No test case chooses the network, checkpoint, ridge, rank or mask parameters.

All 1,797 per-image hidden errors for all five saved method entries are
retained, including the duplicate rank-16 PCA entry. Summary tables expose
four distinct methods. The first ten test indices are rendered for all cases;
no successful examples are filtered in. All four methods use the exact same
mask within a case. Their mean MSE values are descriptive on this fixed
historical test set. One draw per row and a single training seed do not support
a repeated-mask confidence interval. Test targets are used to score, not tune.

At 16 scattered missing entries, network MSE .0353774 is worse than neighbor
.0312412 and PCA .0317004. At 32, network .0340515 is below neighbor .0363968
and PCA .0410443. At 48, .0546866/.0549125/.0547961 are close. Block-16 errors
are .0649021/.0650337/.0723497. Block pixels occur in different positions than
scattered masks; equal hidden counts do not isolate mask geometry or explain
an effect. The four displayed cases are not a broad restoration benchmark.

## Interactive inference and reproducibility

The browser bundles selected weights, PCA mean/basis and 3,062 public fitting
rows. It computes new predictions for arbitrary masks. The reference is drawn
and used to score but hidden positions are erased before all predictors.
Independent scalar JS is checked against native Python on all 40 exported
examples/masks, with absolute tolerance 2e-12. It is not bitwise identical
matrix-multiplication reduction order, exact rational inference or an interval
certificate. UI exposes full JSON, source hashes, clipboard failure and download.

Refits use tolerance rtol=1e-7, atol=1e-8 on selected weights/bases; evaluation
uses tighter rtol=1e-10, atol=1e-12 on fixed models. No cross-BLAS bitwise training
claim. CI limits BLAS threads and pins NumPy 2.2.6. SVG and browser export are
byte-exact transformations of stored JSON; SVG metadata embeds the whole
compressed report and binds hidden rows as well as the visible table and pixels.
The gzip timestamp is zero and descriptive OS header is normalized to 255;
Python 3.11/3.12 otherwise expose a platform OS byte for mtime=0. Payload and CRC
are unchanged. The strict whole-SVG byte comparison remains enabled.

No plausible-looking reconstruction is proof of true missing detail, semantic
correctness, calibrated uncertainty or recoverability. No generative samples,
clinical uses, naturally missing-data guarantee, production performance or
privacy assurance are claimed.

## Sources checked November 8, 2026

- Vincent et al. (2010), [Stacked Denoising Autoencoders](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf), Sections 3.1 and 3.4. Primary denoising and emphasized hidden-coordinate loss. This project is not stacked layerwise pretraining or their benchmark reproduction.
- [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), Alpaydin & Kaynak (1998), DOI 10.24432/C50P49. Dataset, original split, block counts, attribution and current CC BY 4.0 notice.
- Ilin & Raiko (2010), [Practical Approaches to PCA in the Presence of Missing Values](https://jmlr.org/papers/volume11/ilin10a/ilin10a.pdf), observed-coordinate objectives, rank and regularization limits. We use a complete-fitting-data PCA basis and per-query coefficient inference, not their incomplete-training-data estimators.
- [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html), official basis shape/order and LAPACK implementation.
- [NumPy lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html), independent augmented least-squares reference semantics.
- Stanford STATS 306B, [PCA with Missing Data](https://web.stanford.edu/~lmackey/stats306b/doc/stats306b-spring14-lecture17_scribed.pdf), 2014. Teaching reference on observed-entry objectives and conditional ridge subproblems. No source code or scaling result copied.
- Fleuret, [Denoising autoencoders](https://www.fleuret.org/dlc/materials/dlc-slides-7-3-denoising-autoencoders.pdf), teaching context on erasure and block corruption. No source measurement transferred.
- Zhang et al., [A Survey on Masked Autoencoder for Visual Self-supervised Learning](https://www.ijcai.org/proceedings/2023/0762.pdf), IJCAI 2023. Distinguishes broad masked modeling from classical denoising. No ViT/MAE representation or downstream performance is claimed here.
