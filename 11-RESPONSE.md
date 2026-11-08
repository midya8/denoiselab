# Fixed-mask response, not finite-edit attribution

This extension differentiates the output of the existing fixed neural and PCA models with respect to normalized observed pixel values. It does not differentiate missing flags, fit parameters again, read hidden targets, compute a new accuracy score or alter the original test audit. The new page takes the same first ten public test rows and four fixed masks. No examples are filtered for a successful reconstruction or large response.

## Output convention

Rows of every 64×64 Jacobian are output pixels; columns are supplied input pixels. Mask 1 means missing. The encoded input is `concat(where(mask,0,x),mask)`, so hidden supplied-value columns are zero whenever an output derivative is defined. Visible outputs are copied from input, giving identity rows. Derivatives are local normalized-pixel response, not causal importance, semantic evidence, a completeness attribution or an adversarial certificate.

No labels or complete hidden target values enter the response artifact. Both original and edited arrays are erased at hidden positions before they are serialized or passed to models. Hidden coordinates in `delta` are zero. The API accepts multiple observed edits, although the browser edits one coordinate at a time. Changing a mask requires a different query and is not a differential operation in this API.

## Neural chain rule

The existing network has three tanh hidden layers and a sigmoid output. Start with a 128×64 encoded-input Jacobian: the first 64 rows have the visible-coordinate diagonal and the missing-flag rows are zero. If a layer uses row-vector forward convention `a_next = phi(a W + b)`, propagate

`J_next = diag(phi'(preactivation)) W.T J`.

Use `1 - tanh(z)^2` and `sigmoid(z)(1-sigmoid(z))`. Replace visible output rows by the identity. No autodiff is used. Python uses NumPy matrix products; JavaScript independently implements scalar loops. Stored original models, artifact hashes, train/validation/test split and original error table are unchanged.

## Observed-only PCA ridge response

Let B be rank×pixels, O observed coordinates, H missing coordinates, and positive ridge rho. The coefficient system is

`A = B_O B_O.T + rho I`, `a = solve(A, B_O (x_O - mean_O))`.

For this fixed observed set, A is independent of x. Thus

`d a / d x_O = solve(A, B_O)` and `d raw_H / d x_O = B_H.T solve(A, B_O)`.

The code solves a matrix right-hand side in Python, with an explicit solve residual; scalar JavaScript uses independently implemented Cholesky solves. It does not form an explicit inverse. Unclipped output is affine in observed values. A strictly saturated hidden output has zero local derivative. At exact raw 0 or 1, a nonconstant row is marked undefined with a whole null Jacobian row. A constant row at that boundary still has its valid zero derivative. The clipping boundary is tested in the computed binary64 values, not certified exact real arithmetic; near-boundary active sets can be numerically sensitive. Visible outputs remain identity rows.

## Finite changes and control methods

For each smooth method the witness retains original prediction, full Jacobian, actual edited prediction, local tangent `f(x)+J delta`, and actual-minus-tangent residual. The hidden residual L2 norm compares these two model outputs, not either output against a target. A tangent can leave [0,1]; the witness retains it unbounded while the displayed image alone clamps to a grayscale range. A null PCA tangent row is not silently scored as zero.

Nearest-neighbor retrieval has piecewise constant hidden outputs with discontinuous switches. We do not publish a global NN Jacobian. Its third image explicitly freezes the originally retrieved row's hidden values, and its witness reports old/new indices, whether the index changed, and actual-minus-frozen hidden values. The original index is chosen by existing observed-coordinate squared distances and lowest fitting index for a computed tie. The runner-up squared-distance gap is descriptive, not a probability or certified perturbation radius. Arbitrary floating edits are not covered by the original quantized-distance exactness claim; near-boundary platform ties may differ.

The pixel-mean control's hidden output really is constant under observed-value edits. Visible output coordinates copy the edit for both controls.

A scalar discontinuity control uses fitting rows [0,0] and [1,1] and mask [0,1]. At observed first value 1/2 there is a tie, resolved to row 0, so the hidden output is 0. In ideal arithmetic any positive increase switches to row 1 and hidden output 1. Tests use the representable finite edit .5 to .500001. This is an explicit retrieval boundary, not evidence that a trained control is generally unstable or a method-quality comparison.

## Independent checks

The combined suite has 35 Python tests and eight Node/real-Chrome tests. The eight new Python tests cover full small-network input finite differences, every observed column of both saved-model 64×64 Jacobians, a separate augmented least-squares PCA coefficient reference, clipping saturation/boundaries, erased hidden-value equality, visible identity, no-input-mutation, all-hidden/all-visible masks, the nearest switch, invalid inputs and the new CLI.

Three new Node tests compare 120 full native/scalar response witnesses across all 40 original public row/mask cases and three fixed edit values; this includes 983,040 Jacobian entries, full original/edited outputs, tangents, residuals and neighbor indices. A separate scalar finite-difference check verifies every observed input column for the saved neural and PCA models. Scalar controls test clipping, a discontinuous neighbor change and invalid input.

One actual-Chrome test exercises 480 method/row/mask/edit views, independently recomputes their witnesses, inspects 8,192 actual signed-heatmap cell colors across every output coordinate, checks erase semantics, edit/reset, keyboard selection, clipboard success/failure, an actual Blob download and 320/390/1280 responsive overflow/touch targets. UI-to-module agreement is not a third mathematical oracle. Numerical derivatives use declared tolerances; no exact-real Jacobian claim is made.

The installed wheel has eight modules including root and five commands. Its reproduction still refits and evaluates all original models outside the checkout and now also runs the response CLI. Original report and image exports remain byte-exact.

## Run

```sh
python -m denoiselab response --input examples/response.json
```

Input keys are `pixels`, `mask`, `edited`, each a single list of width 64. Pixel values must be finite within [0,1]; mask values use the existing binary semantics. The command loads the unchanged selected models and original fitting rows bundled with the package. No API calls, network inference or private data.

```python
from denoiselab.response import network_jacobian, pca_jacobian, response
# network: Denoiser, pca: PCA; fit: fitting-row matrix
witness = response(network, pca, fit, pixels, mask, edited)
```

## Sources and boundaries

Fetched November 8, 2026. Relevant sections were read, not a claim of entire linked-course or paper review.

- Stanford CS231n, [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/), introduction, local-gradient interpretation, chain rule and sigmoid sections. Standard input chain rule and the distinction between infinitesimal and large finite changes. No source code, course benchmark or interpretation guarantee is imported.
- NumPy, [linalg.solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html), full-rank linear-system and multiple-right-hand-side semantics. The fetched current documentation is not a claim that the pinned runtime NumPy version changed.
- NumPy, [linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html), least-squares semantics used by the independent augmented-system reference.
- The original [METHOD.md](METHOD.md) eight-source ledger still covers the denoising model, observed-coordinate PCA and licensed UCI data. No new data, paper experiment, external model, third-party runtime package or novelty claim is added.

Fixed-mask local response is not true missing-detail recovery, natural missing-data behavior, calibrated uncertainty, causal attribution, semantic preservation, a stability radius or a trained-model ranking. Stronger original controls and losing neural metrics remain visible on the original page. Numerical saturation, clipping branches, tiny derivatives and nearest-distance ties can be sensitive to finite precision.
