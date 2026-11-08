# Exact single-observed-coordinate retrieval paths

DenoiseLab v0.3 adds complete rational nearest-neighbor paths over one normalized observed value `u ∈ [0,1]`, with every other observed value and mask fixed. It does not change a trained model, fit selection, original test score, data split or original browser export. It does not differentiate a mask, use hidden test values, use labels or refit a model.

## Scope first

- Fitting source pixels and original observed query pixels must be exact multiples of 1/16 in [0,1], matching this public block-count dataset. No rounding or rational approximation is performed. Hidden supplied query pixels are erased before this check. The original edited coordinate is also required to be quantized; only its new parameter can be arbitrary supported rational.
- One observed input coordinate varies, all other observed values stay fixed. This is not a simultaneous-input robustness radius or a certificate for another mask.
- Parameter input is an unsigned integer, decimal with a leading digit, or numerator/positive-denominator string, at most 160 characters with at most 80 decimal places, within [0,1]. It is parsed exactly, not through a floating-point value. Exponents, signed values, numeric JSON values and `.5` are refused; use `0.5` or `1/2`.
- Fitting input is bounded to 1..10,000 rows and 1..64 coordinates. The bundled source uses all 3,062 original fitting rows and 64 coordinates. The chosen coordinate must be observed.
- Exact rational distance order is a distinct arithmetic contract from `baselines.nearest` or the original scalar browser inference near a binary64 tie. No numerical-equivalence claim is made at arbitrary rational edits or nearly tied distances. Those original implementations remain unchanged.
- Hidden reconstruction uses the selected fitting row. Visible values copy the edit. This is not hidden-detail truth or a new accuracy result.

## Cancel a common quadratic

Let the candidate's original source count at the varied coordinate be integer `c_i ∈ {0,...,16}`. For every other observed coordinate j, let candidate count be `a_ij` and fixed query count be `x_j`. Define

`b_i = c_i² + sum_{j observed, j != input} (a_ij - x_j)²`.

The normalized squared distance satisfies

`256 d_i(u) = 256 u² - 32 c_i u + b_i`.

The quadratic term is shared by all candidates, so compare only affine lines

`L_i(u) = m_i u + b_i`, with `m_i = -32 c_i`.

Each intercept is an integer between 0 and 16,384 under the supported width/count bounds; every initial integer arithmetic operation is exact in both NumPy int64 and JavaScript Number. Coefficients are converted to arbitrary-precision integers for rational arithmetic. No candidate is discarded based on a floating epsilon.

Candidates with the same c have the same slope. Within each slope, retain the least intercept and **all** rows attaining it. Thus at most 17 representative slopes remain. A strictly larger intercept cannot win anywhere. Identical minimum lines can tie across a whole open interval, even when those rows have different hidden pixels. The selected index is the lowest fitting-array index, not necessarily the smallest original training-row ID.

## Lower envelope and point ties

Process slopes in descending order, equivalent to counts in ascending order. Maintain an ordered hull and the rational parameter where each line becomes lower than the preceding line. For a newly added line n and current last line l, their crossing is

`u = (b_n - b_l) / (m_l - m_n)`.

If that crossing is no later than the last line's start, pop the last line and recompute against the previous line. Otherwise append the new line and crossing. Clip the resulting open-interval partition to [0,1]. This is standard same-curvature squared-distance lower-envelope geometry, not a new nearest-neighbor algorithm or an approximate search method.

Discarding zero-width intervals does not discard their point ties. Evaluate **all original fitting rows** at every finite partition endpoint, including 0 and 1. Retain every exact minimizer and apply the lowest fitting-index rule separately there. A line pruned from the hull can still win a single triple-tie point while winning neither adjacent open interval.

An explicit control uses fitting rows `[[.5,.5], [0,0], [1,0]]`, query `[.5,0]`, both coordinates observed and varying coordinate 0. Row 1 wins `(0,.5)`, row 2 wins `(.5,1)`, and all three tie at `.5`; row 0 wins exactly that point. This prevents an interval-only tie rule from silently dropping a legitimate point winner. Separate controls cover coincident lines, domain-endpoint ties and a single fitting row.

## Verifier and source matching

`verify_path(witness, fit, pixels, mask, input_pixel)` does not regenerate the hull. It receives the intended source independently of the witness and recomputes every row's integer coefficients. It checks source width/order, erased query, mask, input coordinate, all slope groups, row coefficients, interval coverage, strict interval order, nonmaximal duplicate partitions, interval winners and ties, all endpoint tie sets, reported distances, slack and check count. Source recomputation in Python uses direct scalar loops, separately from the builder's vectorized coefficient calculation.

For selected row k on an interval [a,b], every candidate has affine difference

`L_i(u) - L_k(u)`.

The verifier checks this is nonnegative at both a and b for **every** source row. Every interior value is a convex combination of endpoint differences, so no candidate is lower inside. Midpoint evaluation identifies all coincident minimum lines and the lowest-index winner; a nonidentical affine line cannot have zero difference at an interior point while remaining nonnegative on both sides. All-row endpoint evaluation checks the point winners separately. The complete partition starts at 0 and ends at 1 without gaps. Reported dominance checks are `2 × fitting_rows × open_intervals`.

A valid proof of edited coefficients is not a valid proof of the intended original source. Changing the source and rebuilding a self-consistent witness is tested and refused against the original source. Source matching is not authentication, a cryptographic signature or permission to trust an arbitrary externally supplied `fit` array. A caller must supply its intended source, not extract a replacement source from the witness. `query_path` evaluates a witness's row lines without source verification; verify it before relying on it. Both CLIs and the browser perform the source check.

Python uses reduced `fractions.Fraction` values. Browser arithmetic uses BigInt rational numerator/denominator pairs, Euclidean GCD normalization and cross-product comparison. BigInt division never evaluates a fractional branch condition. Public rational values are canonical two-string pairs, e.g. `["17","64"]`. Source coefficients and indices are bounded JSON integers. Browser drawing coordinates and grayscale alone use floating approximations; exact fractions in tables/JSON decide queries and ties.

This verifier is an educational mathematical/source consistency check, not a sandbox or untrusted-document security boundary. It is bounded for the supported inputs and does not claim general parser security or full-source authentication.

## What the browser shows

[The path inspector](https://midya8.github.io/denoiselab/path.html) exposes the same first ten original public test rows and four original masks. Any observed pixel can be selected. No example is ranked by path complexity, successful reconstruction or accuracy. Initial pixel is the observed coordinate nearest index 27, a layout choice, not a best-result search.

The colored bar shows open-interval extents, not probabilities. Exact interval fractions, all endpoint tie sets, selected original fitting-row IDs, all tied original IDs and rational squared distances remain in the text and downloadable witness. Buttons query exact rational interval midpoints and endpoints. An input accepts other supported rational values. Invalid input is explicitly not applied, keeping the last valid result labeled below it. Complete per-row line coefficients and source-checked proof status are retained. Candidate-group details show exact, potentially unreduced rational affine scores at the selected value; they omit only the common quadratic term.

Original observed input, edited input and retrieved reconstruction are separate. Grayscale is explicitly approximate for display; hidden coordinates in the query are blue and erased. Copied visible edits are not presented as predicted missing pixels. No hidden-pixel accuracy is computed. Adjacent selected rows can yield identical hidden images; a retrieval-index switch is not necessarily an output-value jump.

## Independent tests and reproduction

The complete v0.3 suite has **44 Python tests and eleven Node/Chrome tests**. Nine new Python tests include:

- 80 fixed-seed random small quantized datasets, an independently enumerated set of all candidate-pair crossings and interval midpoints, with **5,344 exact rational queries** checked against direct all-row squared distances. This oracle does not reuse builder coefficients, groups, hull or crossing helpers.
- All 40 stored public row/mask cases, selecting the observed coordinate nearest index 27, source checks over all 3,062 fitting rows, independent direct rational distances at 0, 1/3, 1/2 and 1 (**160 queries**), and proof size bounds.
- Coincident lines, endpoint and pruned triple-point ties, all source quantization, erased hidden query values, no input mutation, invalid scope/parameters, source replacement, proof tampering and both CLI paths.

Two new Node tests compare all **40 full Fraction/BigInt path witnesses**, source checks and the same 160 query answers, including every row coefficient, interval, point tie, slope group and exact distance. Scalar controls independently exercise point-only winners, coincident lines, parsing and tamper refusal. Agreement is a cross-language check, not a third proof.

One real-Chrome test exercises **160 public exact queries** over all 40 row/mask cases, checks source status, partition/tie displays, original-index mapping and **30,720 actual canvas cell colors**. It also checks a multi-interval edit, exact point/midpoint buttons, invalid-value retention/reset, keyboard selection, clipboard denial/success, a real Blob download, and 320/390/1280 overflow/touch targets. UI-to-module agreement is not an independent mathematics oracle. Actual phone/desktop rendering is inspected separately.

The installed wheel has **nine modules and seven commands**. The isolated reproduction still performs the original full refit and held-out evaluation, then adds path creation and independent-source path verification. Original model/report/data/browser exports are unchanged.

```sh
python -m denoiselab edit-path --input examples/edit-path.json > /tmp/path.json
```

Path input has exactly `pixels`, `mask`, `input_pixel`, `value`. `value` is a string. Output contains `path`, `verification`, `query`. For `verify-path`, replace the input's `value` with a `witness` containing the path object. Verification loads the original fitting rows from the bundled original source and exits 2 for a mismatch.

```python
from denoiselab.edit_path import edit_path, query_path, verify_path
witness = edit_path(fit, pixels, mask, input_pixel)
check = verify_path(witness, fit, pixels, mask, input_pixel)
answer = query_path(witness, "1/3")
```

## Sources

Fetched November 8, 2026. Relevant sections were read; no claim of reproducing source experiments or reviewing every linked resource.

- Felzenszwalb and Huttenlocher, [Distance Transforms of Sampled Functions](https://theoryofcomputing.org/articles/v008a019/v008a019.pdf), Theory of Computing 8 (2012), pp 415-428. Sections 1.1 and 2.1, Algorithm 1 and its proof: same-shape quadratic lower envelopes and exact intersection algebra. This project adapts the established geometry with explicit tie sets and an independent intended-source check; it does not implement the complete multidimensional transform or claim the paper's benchmarks.
- Python standard library, [fractions](https://docs.python.org/3/library/fractions.html): exact numerator/denominator construction, normalization and the difference between exact binary floats and decimal strings. The fetched current page identifies Python 3.15.2; the project still supports Python 3.10+ and uses long-established APIs, not new formatting methods.
- MDN, [BigInt](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/BigInt): fractional division truncates, integer comparison and JSON serialization constraints. Branch comparisons use cross-products and exported rationals use decimal strings.
- Original [METHOD.md](METHOD.md) retains the UCI data/license, model fitting and strong-control ledger; [RESPONSE.md](RESPONSE.md) covers local derivatives and finite edits. No new dataset, model, external runtime dependency, uncertainty estimate or accuracy claim is added.
