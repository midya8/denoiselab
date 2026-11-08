# Two observed values, complete rational cells

DenoiseLab v0.4 adds a two-coordinate nearest-neighbor slice over `(u,v) ∈ [0,1]²`. Every other observed input, the missing mask and all original fitting rows stay fixed. This is a closed minimum-cell map, not a classifier boundary, natural-image manifold, general robustness radius, causal recourse, uncertainty estimate or new accuracy result. Original trained models, source data, splits and test scores are unchanged.

## Arithmetic and source contract

The original fitting and observed query values must be exact multiples of 1/16 in [0,1], matching the public block-count dataset. Hidden supplied query values are erased before quantization checks; they do not enter coefficients or queries. The ordinary input validator still requires all supplied values to be finite and within [0,1]. Fitting input has 1..10,000 rows and 2..64 columns. Two distinct observed input indices are required. The browser uses all 3,062 original fitting rows and all 64 coordinates.

Each new parameter follows the scalar path contract: an unsigned integer, a decimal with a leading digit, or numerator/positive-denominator string, at most 160 characters with at most 80 decimal places, within [0,1]. Numeric JSON values, exponents, signs and `.5` are refused. Use `0.5` or `1/2`. Fractions are exact and reduced. No epsilon, grid search, rounding of source values or float-based geometric branch is used. Drawing and grayscale alone use float approximations. Binary64 nearest-neighbor inference near a tie is a different numerical contract; exact tie-equivalence is not claimed.

Let the candidate's source counts at the edited coordinates be integers `c_i,d_i ∈ {0,...,16}`. Let other observed source/query counts be `a_ij,x_j`. Set

`b_i = c_i² + d_i² + sum_{other observed j}(a_ij - x_j)²`.

The normalized squared distance obeys

`256 distance_i²(u,v) = 256(u² + v²) - 32 c_i u - 32 d_i v + b_i`.

The common quadratic cancels. Each candidate is represented by an integer affine function `L_i(u,v)`. All source coefficients are bounded exact integers; rational geometry uses Python Fraction or reduced BigInt numerator/denominator pairs. BigInt division only reduces exact integer fractions, never truncates a fractional branch value. JSON rationals are two decimal strings.

## Closed minimum regions, including zero area

Group by the two source counts, yielding at most 17² = 289 slopes. Retain the least intercept in each group and every original fitting index attaining it. A larger intercept at the same slope cannot even tie anywhere. Every original row coefficient remains in `row_lines`; no fitting examples are removed from the source or query. Within a coincident group the representative is the lowest fitting-array index, which need not be the lowest original dataset-row ID.

The region for group k is the closed intersection

`C_k = [0,1]² ∩ { (u,v): L_i(u,v) - L_k(u,v) >= 0 for every group i }`.

This is standard power-diagram / affine lower-envelope geometry. The implementation repeatedly clips a counterclockwise square against one closed halfplane, constructing crossings by `s / (s-e)` using exact signed affine values. Consecutive duplicates and redundant collinear vertices are removed. Segments are stored by ordered endpoints; points by one vertex. An empty result is kept as an explicit empty cell with a checked infeasibility combination. These are not discarded because they have no area.

Every group has one cell with dimension 2, 1, 0 or -1 (empty). Positive-area polygons have disjoint interiors and their exact areas sum to one. Shared edges and points can belong to multiple minimum cells. A cell's group representative need not win on a shared boundary: another group can have a lower fitting index there. Queries therefore evaluate **all original row lines**, retain the entire exact tie set and use its lowest fitting index. The cell list is a map of closed minima, not disjoint tie-broken winner territories.

A synthetic segment-only control has fitting rows `[[.5,0,.5],[0,0,0],[1,0,0]]`, all inputs observed, fixed third query value zero and axes 0,1. Row 0 minimizes only on `u=.5`, for the entire v interval, and wins that tie by index. For a point-only control, use row 0 `[.5,.5,.5,.5]` and the four `(0/1,0/1,0,0)` corners, all observed, fixed last two query values zero. Row 0 minimizes and wins only at `(.5,.5)`. Empty-cell and coincident-line controls are separate.

## A verifier that does not repeat clipping

`verify_plane(witness, fit, pixels, mask, input_pixels)` receives the intended source independently. It recomputes every coefficient, erased input, mask, coordinate order, groups and ties. Python uses direct scalar coefficient sums separately from the builder's NumPy calculation. JavaScript shares its source/coefficient helper between builder and verifier; cross-language agreement is not a third independent proof. Neither implementation calls the clipping routine during verification.

For each cell, the verifier establishes both containment directions:

- **Polygon:** Every rational vertex satisfies every source-group and domain halfplane. The ordered vertices are strictly convex, counterclockwise and without duplicates or redundant collinear edges. Every polygon edge lies on a nonzero original constraint. Thus the proposed polygon is feasible, while the intended feasible set lies inside every one of its supporting edges. The two sets are equal, not merely an arbitrary feasible subpolygon.
- **Segment:** Both endpoints are feasible. Opposing active constraint normals force the entire intended set onto the segment's line. At least one active source/domain constraint bounds each endpoint along that line. The proposed segment is maximal.
- **Point:** The point is feasible. For the constraints active there, every possible boundary ray in either direction is rejected by another active constraint. A nontrivial planar polyhedral feasible-direction cone would contain such a boundary ray. The cone is therefore `{0}`, so the intended set contains only this point. An unconstrained or line-shaped cone is refused.
- **Empty:** A certificate selects one to three source/domain inequalities with nonnegative integer weights. The verifier checks their weighted normal sum is zero and their weighted constant is strictly negative. A feasible point would imply that negative constant is nonnegative, a contradiction. The builder searches constraints active at the last nonempty clipped cell plus the eliminating cut; failure to find a certificate raises an error rather than claiming emptiness.

Source grouping implies the omitted higher-intercept parallel lines cannot beat their minimum group. Every group is checked, including empty groups and lower-dimensional cells. A total area of one is a useful redundant check, not a substitute for these per-cell maximality and infeasibility proofs. Vertex-constraint check counts count representative-group halfplanes plus the four domain constraints, not the cost of every original-row comparison or a speed benchmark.

This is an educational mathematical/source consistency checker, not authentication, a cryptographic signature, a sandbox or an untrusted-document parser security boundary. A caller must supply its intended source, not trust a replacement source embedded in a witness. `query_plane` evaluates lines without source verification: verify first. Both CLIs and the browser verify before displaying a result. These functions do not verify another mask, other edited coordinates, hidden values as targets or general simultaneous-input robustness.

## Browser and reproduction

The [two-pixel inspector](https://midya8.github.io/denoiselab/plane.html) uses the unchanged first ten original test rows and four masks. Initial coordinates are the observed indices nearest 27 and 28, distinct and chosen for layout rather than number of regions, reconstruction quality or accuracy. Later selections are preserved when still valid for the next mask. There is no search for a successful example.

Positive-area polygons are colored; additional segment-only cells are dashed black and point-only cells outlined. Approximate SVG coordinates never decide a query. Clicking the square explicitly snaps each value to a rational multiple of 1/256. Text input and exact sample, vertex and edge-midpoint buttons avoid that grid. The exact vertex coordinates, areas, coincident original row IDs, complete query tie sets, all source lines and every empty proof remain available as text/JSON. Invalid input is not applied and the last valid result stays labeled. Keyboard, clipboard failure/success and actual Blob download are tested. Missing input pixels are blue; visible edits are copied, not described as recovered details. No test label or hidden-pixel score is computed.

The complete suite has **52 Python tests and fourteen Node/Chrome tests**. Eight new Python tests add:

- Independent all-boundary-pair intersection enumeration on 30 fixed-seed eight-row sources, using direct rational squared distances to construct boundaries, with **238 group cells and 2,110 exact queries**. This oracle does not use production coefficients, grouping, clipping, certificate or geometry helpers.
- All **40 original row/mask cases**, axes nearest 27/28, all **3,062 fitting rows**, source verification and **160 independently computed exact squared-distance queries**. Those selected slices contain **8,223 empty, two point-only, two segment-only and 232 area cells**. These counts describe the fixed audit selection, not all possible coordinate pairs.
- Point-only and segment-only winners, domain/corner ties, coincident lines, hidden erasure, no mutation, invalid source/parameter scope, source replacement, geometry/area/empty-proof tampering and both CLI paths with refusal exit codes.

Two new Node tests compare all 40 complete native/Fraction and BigInt witnesses, including every row coefficient, empty combination, polygon/segment/point vertex, area and tie/query answer. Separate controls test degeneracies and refusals. Agreement is cross-language checking, not an independent mathematical oracle. The real-Chrome test covers all forty row/mask cases and **160 actual UI queries / 30,720 canvas-cell colors**, verifies source status and cell/tie displays, exact buttons, keyboard coordinate changes, snapped pointer queries, invalid-value retention, clipboard denial/success, real JSON Blob content and visible 44px controls at 320/390/1280 widths. UI-to-module agreement is not a third geometry oracle; actual pixels are inspected separately.

Installed wheel reproduction imports **ten modules and exercises nine commands**, retaining the original full refit and held-out evaluation.

```sh
python -m denoiselab edit-plane --input examples/edit-plane.json > /tmp/plane.json
```

Input has exactly `pixels`, `mask`, `input_pixels`, `values`. The last two are two-element arrays of distinct observed integer indices and parameter strings. Output has `plane`, `verification`, `query`. For `verify-plane`, replace `values` with `witness` containing the plane object. The CLI loads original fitting rows from the bundled source and exits 2 on a mismatch.

```python
from denoiselab.edit_plane import edit_plane, verify_plane, query_plane
w = edit_plane(fit, pixels, mask, [26, 29])
checked = verify_plane(w, fit, pixels, mask, [26, 29])
answer = query_plane(w, "1/3", "2/3")
```

## Source notes

Read November 8, 2026. This implements the stated algebra and clipping from scratch; it does not copy source implementations, reproduce paper benchmarks or claim a new power-diagram algorithm.

1. F. Aurenhammer, [Power Diagrams: Properties, Algorithms and Applications](https://www.cs.jhu.edu/~misha/Spring20/Aurenhammer87.pdf), SIAM Journal on Computing 16(1), February 1987. Introduction, section 2 definitions and section 4.1 transform: squared-distance weights, affine separators, convex/empty power cells and their lower-dimensional faces. No complexity claim from its efficient algorithms is applied to this repeated-clip implementation.
2. David Eberly, [Clip a Convex Polygon by a Hyperplane](https://www.geometrictools.com/Documentation/ClipConvexPolygonByHyperplane.pdf), Geometric Tools, created 2019, modified 2020, CC BY 4.0. Introduction and sections 2-4: sign classification, crossing construction and floating-point degeneracies. Our branches use exact rational arithmetic and explicitly support collapsed results.
3. Ivan E. Sutherland and Gary W. Hodgman, [Reentrant polygon clipping](https://cacm.acm.org/research/reentrant-polygon-clipping/), Communications of the ACM, January 1, 1974, pp. 32-42. Publisher abstract read, not the full paper: clipping successively against individual convex boundaries with ordered output vertices. The search result's newer date is not the paper's publication date.
4. Yinyu Ye, [Midterm Review](https://web.stanford.edu/class/msande310/midtermreview09.pdf), Stanford Management Science and Engineering. Farkas lemma and alternative-system section, especially Theorem 3: nonnegative combinations with zero normal and a negative constant certify inequality infeasibility. The exact combination is verified directly here; no external solver is used.
5. Python, [fractions](https://docs.python.org/3/library/fractions.html): exact construction, normalization and binary-float versus decimal-string distinction. Current fetched page says Python 3.15.2; the implementation uses long-established APIs and supports Python 3.10+.
6. MDN, [BigInt](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/BigInt): integer division truncation, comparisons and JSON constraints. Rational comparisons use cross-products; exported values are strings.

[METHOD.md](METHOD.md), [RESPONSE.md](RESPONSE.md) and [PATH.md](PATH.md) retain original data/license, fitting, derivative and scalar-path sources and limits.
