/**
 * @file
 * Header: GMM-based Segmentation Engine
 *
 * Implements interactive foreground/background segmentation using Gaussian
 * Mixture Models (GMMs). The user paints scribbles to indicate definite
 * foreground and background regions; the algorithm then classifies all
 * remaining pixels by comparing their colour likelihood under the two models.
 *
 * Inputs:
 *   - A QImage (RGB or greyscale) — the source tomogram slice
 *   - A trimap QByteArray: one byte per pixel with values
 *       TRIMAP_BACKGROUND  — forced background (user-painted red scribble)
 *       TRIMAP_FOREGROUND  — forced foreground (user-painted green scribble)
 *       TRIMAP_UNKNOWN     — to be determined by the algorithm
 *
 * Output:
 *   - A per-pixel alpha mask: ALPHA_FG (255) or ALPHA_BG (0)
 *   - The fitted foreground and background GMMs (for propagation to adjacent slices)
 *
 * The output alpha mask is written directly into the SPIERSedit GA[] greyscale
 * image for the current segment, integrating with the existing threshold system
 * (pixels >= 128 are treated as fossil). Foreground pixels get 255, background 0.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2026 by the SPIERS contributors.
 */

#ifndef GRABCUT_H
#define GRABCUT_H

#include "gmm.h"

#include <QByteArray>
#include <QImage>

#include <functional>
#include <vector>

// ─── Trimap label values ──────────────────────────────────────────────────────

static constexpr uchar TRIMAP_BACKGROUND = 0;   ///< Definite background
static constexpr uchar TRIMAP_FOREGROUND = 255;  ///< Definite foreground
static constexpr uchar TRIMAP_UNKNOWN    = 128;  ///< Unknown — algorithm decides

// ─── Alpha mask values ────────────────────────────────────────────────────────

static constexpr uchar ALPHA_BG = 0;   ///< Background pixel in output GA[]
static constexpr uchar ALPHA_FG = 255; ///< Foreground pixel in output GA[]

/**
 * @brief Encapsulates the full state of a segmentation session for one slice.
 *
 * Holds the fitted GMMs, the per-pixel component assignments, and the current
 * alpha mask. Can be serialised for persistence and used as a warm-start seed
 * when propagating to adjacent slices.
 */
struct GrabCutState
{
    GMM bgGmm;  ///< Background Gaussian Mixture Model (K=5 components)
    GMM fgGmm;  ///< Foreground Gaussian Mixture Model (K=5 components)

    std::vector<uchar> alpha;       ///< Current segmentation: ALPHA_FG or ALPHA_BG
    std::vector<int> componentMap;  ///< Per-pixel GMM component assignment (0..K-1)

    bool isInitialised = false; ///< True once GMMs have been seeded from a trimap

    void clear()
    {
        isInitialised = false;
        alpha.clear();
        componentMap.clear();
    }
};

/**
 * @brief GMM-based segmentation engine.
 *
 * Typical usage for a new slice:
 * @code
 *   GrabCut gc;
 *   gc.setImage(colourImage);
 *   gc.setTrimap(trimap);
 *   gc.initialise();
 *   gc.run(1);
 *   QByteArray result = gc.alphaAsGAImage(fwidth4);
 *   GrabCutState savedState = gc.getState();
 * @endcode
 *
 * Warm-start from a propagated state (adjacent slice):
 * @code
 *   gc.setImage(newImage);
 *   gc.setTrimap(newTrimap);
 *   gc.setState(propagatedState);
 *   gc.run(1);
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
     * @param img  Source image.
     */
    void setImage(const QImage &img);

    /**
     * @brief Set the trimap defining forced fg/bg regions and the unknown area.
     * @param trimapData  One byte per pixel (row-major). Use TRIMAP_* constants.
     */
    void setTrimap(const QByteArray &trimapData);

    /**
     * @brief Restore from a previously saved state (e.g. from an adjacent slice).
     * @param savedState  State to restore from.
     */
    void setState(const GrabCutState &savedState);

    /**
     * @brief Optional progress callback, called with integer 0-100 during run().
     * @param cb  Callback function.
     */
    void setProgressCallback(std::function<void(int)> cb) { progressCb = cb; }

    // ── Algorithm steps ───────────────────────────────────────────────────────

    /**
     * @brief Seed the foreground and background GMMs from the trimap scribbles.
     *        Must be called before run() when starting on a fresh slice.
     */
    void initialise();

    /**
     * @brief Run the segmentation. Classifies all unknown pixels by comparing
     *        their colour likelihood under the foreground and background GMMs.
     * @param iterations  Number of iterations (GMM refit cycles). 1 is typically
     *                    sufficient; more iterations refine using the output.
     */
    void run(int iterations = 1);

    /**
     * @brief Run a single iteration: refit GMMs from current alpha then reclassify.
     */
    void runOneIteration();

    // ── Output ────────────────────────────────────────────────────────────────

    /**
     * @brief Return the current alpha mask (size = width * height).
     *        Each element is ALPHA_FG (255) or ALPHA_BG (0).
     */
    const std::vector<uchar> &alpha() const { return state.alpha; }

    /**
     * @brief Return the alpha mask formatted for writing into GA[seg].
     *        Respects the fwidth4 row stride.
     * @param fwidth4  Padded row stride used by SPIERSedit's GA[] QImage.
     */
    QByteArray alphaAsGAImage(int fwidth4) const;

    /**
     * @brief Get the current algorithm state for persistence or propagation.
     */
    GrabCutState getState() const { return state; }

    /**
     * @brief Image width (valid after setImage()).
     */
    int width() const { return imageWidth; }

    /**
     * @brief Image height (valid after setImage()).
     */
    int height() const { return imageHeight; }

private:
    int imageHeight = 0;  ///< Image height in pixels
    int imageWidth = 0;   ///< Image width in pixels

    GrabCutState state; ///< Current algorithm state (GMMs + alpha + componentMap)

    std::function<void(int)> progressCb; ///< Optional progress callback

    std::vector<double> pixels; ///< Flat RGB triples normalised to 0-1, size W*H*3
    std::vector<uchar> trimap;  ///< Trimap (one byte per pixel, flat row-major)

    /**
     * @brief Classify each unknown pixel by comparing fg and bg GMM likelihoods.
     *        Trimap-forced pixels are always honoured.
     */
    void classifyByGmm();

    /**
     * @brief Get normalised RGB for pixel index i.
     */
    void getPixel(int i, double &r, double &g, double &b) const
    {
        r = pixels[static_cast<size_t>(i) * 3 + 0];
        g = pixels[static_cast<size_t>(i) * 3 + 1];
        b = pixels[static_cast<size_t>(i) * 3 + 2];
    }

    /**
     * @brief Initialise alpha from the trimap.
     */
    void initAlphaFromTrimap();

    /**
     * @brief Convert (x, y) to flat index.
     */
    int pixelIndex(int x, int y) const { return y * imageWidth + x; }
};

#endif // GRABCUT_H
