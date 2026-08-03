/**
 * @file
 * Header: GrabCut Segmentation Algorithm
 *
 * Implements the GrabCut algorithm as described in:
 *   Rother, Kolmogorov & Blake, "GrabCut: Interactive Foreground Extraction
 *   using Iterated Graph Cuts", SIGGRAPH 2004.
 *
 * The algorithm iterates between:
 *   1. Assigning each pixel to a GMM component (E-step)
 *   2. Re-estimating GMM parameters from those assignments (M-step)
 *   3. Minimising a Gibbs energy via graph cut to update the alpha (fg/bg) mask
 *
 * Inputs:
 *   - A QImage (RGB or greyscale) — the source tomogram slice
 *   - A trimap QByteArray: one byte per pixel with values
 *       TRIMAP_BACKGROUND  — forced background (user-painted red scribble)
 *       TRIMAP_FOREGROUND  — forced foreground (user-painted green scribble)
 *       TRIMAP_UNKNOWN     — to be determined by the algorithm
 *
 * Output:
 *   - A QByteArray alpha mask: ALPHA_FG (255) or ALPHA_BG (0) per pixel
 *   - The fitted foreground and background GMMs (for propagation to adj. slices)
 *
 * The output alpha mask is written directly into the SPIERSedit GA[] greyscale
 * image for the current segment, so it integrates with the existing threshold
 * system (pixels >= 128 are fossil). FG pixels get value 255, BG pixels get 0.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2024 by the SPIERS contributors.
 */

#ifndef GRABCUT_H
#define GRABCUT_H

#include <QImage>
#include <QByteArray>
#include <vector>
#include <functional>
#include "gmm.h"
#include "maxflow.h"

// ─── Trimap label values ─────────────────────────────────────────────────────

static constexpr uchar TRIMAP_BACKGROUND = 0;   ///< Definite background
static constexpr uchar TRIMAP_FOREGROUND = 255;  ///< Definite foreground
static constexpr uchar TRIMAP_UNKNOWN    = 128;  ///< Unknown — algorithm decides

// ─── Alpha mask values ────────────────────────────────────────────────────────

static constexpr uchar ALPHA_BG = 0;    ///< Background pixel in output GA[]
static constexpr uchar ALPHA_FG = 255;  ///< Foreground pixel in output GA[]

/**
 * @brief Encapsulates the full state of a GrabCut session for one slice.
 *
 * Holds the fitted GMMs, the per-pixel component assignments, and the current
 * alpha mask. Can be serialised for persistence and used as a warm-start seed
 * when propagating to adjacent slices.
 */
struct GrabCutState
{
    GMM fgGMM;  ///< Foreground Gaussian Mixture Model (K=5 components)
    GMM bgGMM;  ///< Background Gaussian Mixture Model (K=5 components)

    /// Per-pixel GMM component assignment (0..K-1), size = width*height
    std::vector<int> componentMap;

    /// Current segmentation: ALPHA_FG or ALPHA_BG, size = width*height
    std::vector<uchar> alpha;

    bool isInitialised = false;

    void clear() { isInitialised = false; componentMap.clear(); alpha.clear(); }
};

/**
 * @brief The GrabCut engine.
 *
 * Typical usage for a new slice:
 * @code
 *   GrabCut gc;
 *   gc.setImage(colourImage);
 *   gc.setTrimap(trimap);
 *   gc.initialise();          // seeds GMMs from trimap scribbles
 *   gc.run(5);                // 5 iterations
 *   QByteArray result = gc.alphaAsGAImage();  // write into GA[]
 *   GrabCutState state = gc.getState();       // save for propagation
 * @endcode
 *
 * Warm-start from a propagated state (adjacent slice):
 * @code
 *   gc.setImage(newImage);
 *   gc.setTrimap(newTrimap);  // may be empty/unknown if no new scribbles
 *   gc.setState(propagatedState);
 *   gc.run(3);
 * @endcode
 *
 * Progress callback (optional, for UI progress bar):
 * @code
 *   gc.setProgressCallback([](int pct){ myProgressBar->setValue(pct); });
 * @endcode
 */
class GrabCut
{
public:
    GrabCut();

    // ── Input setters ─────────────────────────────────────────────────────────

    /**
     * @brief Set the source colour image for this slice.
     *        Accepts RGB32, ARGB32, Indexed8 or Grayscale8 QImages.
     *        Greyscale images are treated as R=G=B.
     */
    void setImage(const QImage &img);

    /**
     * @brief Set the trimap defining forced fg/bg scribbles and unknown region.
     * @param trimap  One byte per pixel (row-major). Use TRIMAP_* constants.
     *                Must match image dimensions.
     */
    void setTrimap(const QByteArray &trimap);

    /**
     * @brief Restore from a previously saved state (e.g. from adjacent slice).
     *        GMMs are reused; alpha is reinitialised from the new trimap.
     */
    void setState(const GrabCutState &state);

    /**
     * @brief Optional progress callback. Called with integer 0–100 during run().
     */
    void setProgressCallback(std::function<void(int)> cb) { progressCb = cb; }

    // ── Algorithm steps ───────────────────────────────────────────────────────

    /**
     * @brief Seed the foreground and background GMMs from the trimap scribbles.
     *        Must be called before run() on a fresh (non-warm-started) slice.
     */
    void initialise();

    /**
     * @brief Run N full GrabCut iterations (GMM fit + graph cut).
     * @param iterations  Number of iterations. 5 is typically sufficient.
     */
    void run(int iterations = 5);

    /**
     * @brief Run a single GrabCut iteration (one GMM refit + one graph cut).
     *        Useful for incremental/interactive updates after new scribbles.
     */
    void runOneIteration();

    // ── Output ────────────────────────────────────────────────────────────────

    /**
     * @brief Return the current alpha mask as a flat byte array (size = W*H).
     *        Each byte is ALPHA_FG (255) or ALPHA_BG (0).
     *        Forced fg/bg trimap regions are always honoured.
     */
    const std::vector<uchar> &alpha() const { return state_.alpha; }

    /**
     * @brief Return the alpha mask formatted as a QByteArray suitable for
     *        writing directly into GA[seg] (the SPIERSedit greyscale image).
     *        Output is row-major, one byte per pixel, 255=fossil, 0=background.
     *        Respects fwidth4 padding (stride = ((width+3)/4)*4).
     * @param fwidth4  The padded row stride used by SPIERSedit's QImage (GA[]).
     */
    QByteArray alphaAsGAImage(int fwidth4) const;

    /**
     * @brief Get the current full algorithm state for persistence/propagation.
     */
    GrabCutState getState() const { return state_; }

    /**
     * @brief Image width and height (set after setImage()).
     */
    int width()  const { return W; }
    int height() const { return H; }

    // ── Beta (smoothness) parameter ───────────────────────────────────────────

    /**
     * @brief Smoothness weight gamma for the N-link Potts term.
     *        Default 50.0 — as in the original Rother et al. paper.
     */
    double gamma = 50.0;

private:
    // Image data stored as flat double triples [R,G,B] normalised to 0–1
    std::vector<double> pixels;  // size = W*H*3
    int W = 0;
    int H = 0;

    // Trimap (one byte per pixel, flat row-major)
    std::vector<uchar> trimap;

    // Algorithm state
    GrabCutState state_;

    // Precomputed N-link weights (4-connected grid: right, down, right-down, right-up)
    // Stored as parallel arrays indexed by pixel i for neighbour direction d
    // Directions: 0=right, 1=down, 2=diag-down-right, 3=diag-up-right
    struct NLinkWeights
    {
        std::vector<double> right, down, diagDR, diagUR;
    } nlinks;

    // Beta: contrast-normalisation factor for N-links
    double beta = 0.0;

    // Progress callback
    std::function<void(int)> progressCb;

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Precompute beta = 1 / (2 * mean squared colour difference between neighbours)
    void computeBeta();

    /// Precompute all N-link weights from image gradients
    void computeNLinks();

    /// N-link weight between pixels at distance vector (dx,dy)
    double nLinkWeight(int x1, int y1, int x2, int y2) const;

    /// Initialise alpha from trimap: TRIMAP_FG→ALPHA_FG, else→ALPHA_BG
    void initAlphaFromTrimap();

    /// Step 1 of each GrabCut iteration: assign each unknown pixel to a GMM component
    void assignGMMComponents();

    /// Step 2: refit GMMs from current component assignments
    void refitGMMs();

    /// Step 3: graph cut to update alpha
    void graphCut();

    /// Inline: pixel index (row-major)
    int idx(int x, int y) const { return y * W + x; }

    /// Get normalised RGB for pixel i
    void getPixel(int i, double &r, double &g, double &b) const
    {
        r = pixels[static_cast<size_t>(i) * 3 + 0];
        g = pixels[static_cast<size_t>(i) * 3 + 1];
        b = pixels[static_cast<size_t>(i) * 3 + 2];
    }

    /// T-link capacity: -log P(pixel | GMM)  (clamped to avoid inf)
    double dataTermFG(int i) const;
    double dataTermBG(int i) const;

    static constexpr double LOG_CLAMP = 1e-10;  ///< Prevents log(0)
};

#endif // GRABCUT_H
