/**
 * @file
 * Source: GrabCut Segmentation Algorithm
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2026 by the SPIERS contributors.
 *
 * Implementation follows Rother, Kolmogorov & Blake (2004) closely:
 *
 *   Energy:  E(alpha, k, theta, z) =
 *                U(alpha, k, theta, z)   [unary / data term]
 *              + V(alpha, z)             [pairwise / smoothness term]
 *
 *   U = sum_n  -log[ pi_{k_n} * N(z_n | mu_{k_n}, Sigma_{k_n}) ]
 *                   for the appropriate GMM (fg or bg) given alpha_n
 *
 *   V = gamma * sum_{(m,n) in neighbourhood}
 *              [alpha_m != alpha_n] * exp(-beta * ||z_m - z_n||^2) / dist(m,n)
 *
 *   beta = 1 / (2 * <||z_m - z_n||^2>)   (contrast-normalisation)
 *
 * The graph-cut step (step 3) minimises E over alpha holding k and theta fixed.
 * Steps 1–3 are iterated until convergence.
 */

#include "grabcut.h"
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <QtGlobal>

// ─── Constructor ──────────────────────────────────────────────────────────────

GrabCut::GrabCut() {}

// ─── Input setters ────────────────────────────────────────────────────────────

void GrabCut::setImage(const QImage &img)
{
    // Convert to a consistent internal format
    QImage src;
    if (img.format() == QImage::Format_Indexed8
        || img.format() == QImage::Format_Grayscale8)
    {
        src = img.convertToFormat(QImage::Format_RGB32);
    }
    else
    {
        src = img.convertToFormat(QImage::Format_RGB32);
    }

    imageWidth = src.width();
    imageHeight = src.height();
    int N = imageWidth * imageHeight;
    pixels.resize(static_cast<size_t>(N) * 3);

    const uchar *bits = src.bits();
    for (int y = 0; y < imageHeight; y++)
    {
        for (int x = 0; x < imageWidth; x++)
        {
            int i = y * imageWidth + x;
            // QImage::Format_RGB32 layout (little-endian): B G R A
            int byteOffset = (y * src.bytesPerLine()) + x * 4;
            double b = static_cast<double>(bits[byteOffset + 0]) / 255.0;
            double g = static_cast<double>(bits[byteOffset + 1]) / 255.0;
            double r = static_cast<double>(bits[byteOffset + 2]) / 255.0;
            pixels[static_cast<size_t>(i) * 3 + 0] = r;
            pixels[static_cast<size_t>(i) * 3 + 1] = g;
            pixels[static_cast<size_t>(i) * 3 + 2] = b;
        }
    }

    // Precompute beta and N-link weights for this image
    computeBeta();
    computeNLinks();
}

void GrabCut::setTrimap(const QByteArray &tm)
{
    trimap.resize(static_cast<size_t>(imageWidth * imageHeight));
    for (int i = 0; i < imageWidth * imageHeight; i++)
        trimap[static_cast<size_t>(i)] = static_cast<uchar>(tm.at(i));
}

void GrabCut::setState(const GrabCutState &savedState)
{
    state = savedState;
    // Re-honour the trimap: forced regions override the propagated alpha
    for (int i = 0; i < imageWidth * imageHeight; i++)
    {
        if (trimap[static_cast<size_t>(i)] == TRIMAP_FOREGROUND)
            state.alpha[static_cast<size_t>(i)] = ALPHA_FG;
        else if (trimap[static_cast<size_t>(i)] == TRIMAP_BACKGROUND)
            state.alpha[static_cast<size_t>(i)] = ALPHA_BG;
    }
    state.isInitialised = true;
}

// ─── Beta and N-link precomputation ──────────────────────────────────────────

void GrabCut::computeBeta()
{
    // beta = 1 / (2 * <||z_m - z_n||^2>) over all horizontal + vertical pairs
    double sumDist = 0.0;
    int    count   = 0;

    for (int y = 0; y < imageHeight; y++)
    {
        for (int x = 0; x < imageWidth; x++)
        {
            int i = pixelIndex(x, y);
            double r1;
            double g1;
            double b1;
            getPixel(i, r1, g1, b1);

            // Right neighbour
            if (x + 1 < imageWidth)
            {
                double r2;
                double g2;
                double b2;
                getPixel(pixelIndex(x + 1, y), r2, g2, b2);
                double dr = r1 - r2;
                double dg = g1 - g2;
                double db = b1 - b2;
                sumDist += dr * dr + dg * dg + db * db;
                count++;
            }
            // Down neighbour
            if (y + 1 < imageHeight)
            {
                double r2;
                double g2;
                double b2;
                getPixel(pixelIndex(x, y + 1), r2, g2, b2);
                double dr = r1 - r2;
                double dg = g1 - g2;
                double db = b1 - b2;
                sumDist += dr * dr + dg * dg + db * db;
                count++;
            }
        }
    }

    if (count > 0 && sumDist > 0.0)
        beta = 1.0 / (2.0 * sumDist / count);
    else
        beta = 0.0;
}

double GrabCut::nLinkWeight(int x1, int y1, int x2, int y2) const
{
    // Weight = gamma * exp(-beta * ||z1 - z2||^2) / dist(p1, p2)
    double r1;
    double g1;
    double b1;
    double r2;
    double g2;
    double b2;
    getPixel(pixelIndex(x1, y1), r1, g1, b1);
    getPixel(pixelIndex(x2, y2), r2, g2, b2);

    double dr = r1 - r2;
    double dg = g1 - g2;
    double db = b1 - b2;
    double dist2 = dr * dr + dg * dg + db * db;

    // Geometric distance between pixel centres
    int dx = x2 - x1;
    int dy = y2 - y1;
    double geoDist = std::sqrt(static_cast<double>(dx * dx + dy * dy));

    return (gamma / geoDist) * std::exp(-beta * dist2);
}

void GrabCut::computeNLinks()
{
    int N = imageWidth * imageHeight;
    nlinks.right.assign(static_cast<size_t>(N), 0.0);
    nlinks.down.assign(static_cast<size_t>(N),  0.0);
    nlinks.diagDR.assign(static_cast<size_t>(N), 0.0);
    nlinks.diagUR.assign(static_cast<size_t>(N), 0.0);

    for (int y = 0; y < imageHeight; y++)
    {
        for (int x = 0; x < imageWidth; x++)
        {
            int i = pixelIndex(x, y);
            if (x + 1 < imageWidth)
                nlinks.right[static_cast<size_t>(i)] = nLinkWeight(x, y, x + 1, y);
            if (y + 1 < imageHeight)
                nlinks.down[static_cast<size_t>(i)] = nLinkWeight(x, y, x, y + 1);
            if (x + 1 < imageWidth && y + 1 < imageHeight)
                nlinks.diagDR[static_cast<size_t>(i)] = nLinkWeight(x, y, x + 1, y + 1);
            if (x + 1 < imageWidth && y - 1 >= 0)
                nlinks.diagUR[static_cast<size_t>(i)] = nLinkWeight(x, y, x + 1, y - 1);
        }
    }
}

// ─── Data terms ───────────────────────────────────────────────────────────────

double GrabCut::dataTermFG(int i) const
{
    double r;
    double g;
    double b;
    getPixel(i, r, g, b);
    double p = state.fgGmm.probability(r, g, b);
    if (p < LOG_CLAMP) p = LOG_CLAMP;
    return -std::log(p);
}

double GrabCut::dataTermBG(int i) const
{
    double r;
    double g;
    double b;
    getPixel(i, r, g, b);
    double p = state.bgGmm.probability(r, g, b);
    if (p < LOG_CLAMP) p = LOG_CLAMP;
    return -std::log(p);
}

// ─── Initialisation ───────────────────────────────────────────────────────────

void GrabCut::initAlphaFromTrimap()
{
    int N = imageWidth * imageHeight;
    state.alpha.resize(static_cast<size_t>(N));
    state.componentMap.resize(static_cast<size_t>(N), 0);

    for (int i = 0; i < N; i++)
    {
        uchar t = trimap[static_cast<size_t>(i)];
        if (t == TRIMAP_FOREGROUND)
            state.alpha[static_cast<size_t>(i)] = ALPHA_FG;
        else
            // Unknown and background both start as background.
            // The graph cut will update UNKNOWN pixels.
            state.alpha[static_cast<size_t>(i)] = ALPHA_BG;
    }
}

void GrabCut::initialise()
{
    initAlphaFromTrimap();

    int N = imageWidth * imageHeight;

    // Collect FG and BG pixels from trimap
    std::vector<double> fgPixels, bgPixels;
    fgPixels.reserve(static_cast<size_t>(N) * 3);
    bgPixels.reserve(static_cast<size_t>(N) * 3);

    for (int i = 0; i < N; i++)
    {
        double r;
        double g;
        double b;
        getPixel(i, r, g, b);
        uchar t = trimap[static_cast<size_t>(i)];
        if (t == TRIMAP_FOREGROUND)
        {
            fgPixels.push_back(r);
            fgPixels.push_back(g);
            fgPixels.push_back(b);
        }
        else
        {
            // Both background and unknown seed the background GMM initially
            bgPixels.push_back(r);
            bgPixels.push_back(g);
            bgPixels.push_back(b);
        }
    }

    int nFG = static_cast<int>(fgPixels.size()) / 3;
    int nBG = static_cast<int>(bgPixels.size()) / 3;

    if (nFG > 0) state.fgGmm.initFromSamples(fgPixels, nFG);
    if (nBG > 0) state.bgGmm.initFromSamples(bgPixels, nBG);

    state.isInitialised = true;
}

// ─── Algorithm steps ──────────────────────────────────────────────────────────

/**
 * @brief Step 1: For each UNKNOWN pixel, find the most likely GMM component
 *        given the current alpha (fg/bg assignment).
 */
void GrabCut::assignGMMComponents()
{
    int N = imageWidth * imageHeight;
    for (int i = 0; i < N; i++)
    {
        double r;
        double g;
        double b;
        getPixel(i, r, g, b);

        if (state.alpha[static_cast<size_t>(i)] == ALPHA_FG)
            state.componentMap[static_cast<size_t>(i)] =
                state.fgGmm.mostLikelyComponent(r, g, b);
        else
            state.componentMap[static_cast<size_t>(i)] =
                state.bgGmm.mostLikelyComponent(r, g, b);
    }
}

/**
 * @brief Step 2: Refit fg and bg GMMs from the current assignments.
 */
void GrabCut::refitGMMs()
{
    int N = imageWidth * imageHeight;

    state.fgGmm.resetAccumulators();
    state.bgGmm.resetAccumulators();

    for (int i = 0; i < N; i++)
    {
        double r;
        double g;
        double b;
        getPixel(i, r, g, b);
        int comp = state.componentMap[static_cast<size_t>(i)];

        if (state.alpha[static_cast<size_t>(i)] == ALPHA_FG)
            state.fgGmm.addSample(comp, r, g, b);
        else
            state.bgGmm.addSample(comp, r, g, b);
    }

    state.fgGmm.learnFromAccumulated();
    state.bgGmm.learnFromAccumulated();
}

/**
 * @brief Step 3: Build and solve the graph cut to update alpha.
 *
 * Graph structure:
 *   Source (S) = foreground terminal
 *   Sink   (T) = background terminal
 *
 *   T-links (unary):
 *     UNKNOWN pixel i:
 *       S→i  cap = -log P(z_i | bg_GMM)   [cost of calling it background]
 *       i→T  cap = -log P(z_i | fg_GMM)   [cost of calling it foreground]
 *     TRIMAP_FOREGROUND pixel i:
 *       S→i  cap = INF  (forced fg)
 *       i→T  cap = 0
 *     TRIMAP_BACKGROUND pixel i:
 *       S→i  cap = 0
 *       i→T  cap = INF  (forced bg)
 *
 *   N-links (pairwise, 8-connected):
 *       i↔j  cap = V(i,j) = gamma * exp(-beta * ||z_i - z_j||^2) / dist(i,j)
 *
 *   After min-cut:
 *     source side → ALPHA_FG
 *     sink   side → ALPHA_BG
 */
void GrabCut::graphCut()
{
    int N = imageWidth * imageHeight;
    MaxFlowGraph graph(N);

    // Largest observed data term (used for INF-equivalent forced terminal caps)
    const double FORCED = MaxFlowGraph::INF;

    // ── T-links ───────────────────────────────────────────────────────────────
    for (int i = 0; i < N; i++)
    {
        uchar t = trimap[static_cast<size_t>(i)];
        if (t == TRIMAP_FOREGROUND)
        {
            graph.addTLink(i, FORCED, 0.0);
        }
        else if (t == TRIMAP_BACKGROUND)
        {
            graph.addTLink(i, 0.0, FORCED);
        }
        else // TRIMAP_UNKNOWN
        {
            double costFG = dataTermFG(i);  // -log P(z | fg) → cost of fg
            double costBG = dataTermBG(i);  // -log P(z | bg) → cost of bg
            // S→i cap = costBG (cutting S→i means assigning to bg, cost of bg)
            // i→T cap = costFG (cutting i→T means assigning to fg, cost of fg)
            graph.addTLink(i, costBG, costFG);
        }
    }

    // ── N-links (8-connected) ─────────────────────────────────────────────────
    for (int y = 0; y < imageHeight; y++)
    {
        for (int x = 0; x < imageWidth; x++)
        {
            int i = pixelIndex(x, y);

            // Right (horizontal)
            if (x + 1 < imageWidth)
            {
                int j = pixelIndex(x + 1, y);
                double w = nlinks.right[static_cast<size_t>(i)];
                graph.addNLink(i, j, w);
            }
            // Down (vertical)
            if (y + 1 < imageHeight)
            {
                int j = pixelIndex(x, y + 1);
                double w = nlinks.down[static_cast<size_t>(i)];
                graph.addNLink(i, j, w);
            }
            // Diagonal down-right
            if (x + 1 < imageWidth && y + 1 < imageHeight)
            {
                int j = pixelIndex(x + 1, y + 1);
                double w = nlinks.diagDR[static_cast<size_t>(i)];
                graph.addNLink(i, j, w);
            }
            // Diagonal up-right (= down-left of the pixel above)
            if (x + 1 < imageWidth && y - 1 >= 0)
            {
                int j = pixelIndex(x + 1, y - 1);
                double w = nlinks.diagUR[static_cast<size_t>(i)];
                graph.addNLink(i, j, w);
            }
        }
    }

    // ── Solve ─────────────────────────────────────────────────────────────────
    graph.maxflow();
    graph.bfsMarkSource();

    // ── Read back alpha ───────────────────────────────────────────────────────
    for (int i = 0; i < N; i++)
    {
        uchar t = trimap[static_cast<size_t>(i)];
        if (t == TRIMAP_FOREGROUND)
        {
            state.alpha[static_cast<size_t>(i)] = ALPHA_FG;
        }
        else if (t == TRIMAP_BACKGROUND)
        {
            state.alpha[static_cast<size_t>(i)] = ALPHA_BG;
        }
        else
        {
            // Source side = foreground (S is the fg terminal)
            state.alpha[static_cast<size_t>(i)] =
                graph.inSourceSet(i) ? ALPHA_FG : ALPHA_BG;
        }
    }
}

// ─── Public run interface ─────────────────────────────────────────────────────

void GrabCut::runOneIteration()
{
    if (!state.isInitialised)
        initialise();

    assignGMMComponents();
    refitGMMs();
    graphCut();
}

void GrabCut::run(int iterations)
{
    if (!state.isInitialised)
        initialise();

    for (int iter = 0; iter < iterations; iter++)
    {
        if (progressCb)
            progressCb(static_cast<int>(100.0 * iter / iterations));

        assignGMMComponents();
        refitGMMs();

        if (progressCb)
            progressCb(static_cast<int>(100.0 * (iter + 0.5) / iterations));

        graphCut();
    }

    if (progressCb)
        progressCb(100);
}

// ─── Output ───────────────────────────────────────────────────────────────────

QByteArray GrabCut::alphaAsGAImage(int fwidth4) const
{
    // GA images in SPIERSedit use Format_Indexed8 with a stride of fwidth4.
    // The display pipeline renders them bottom-up (inverted Y), so we need to
    // write the alpha data in the same orientation as the source colour image
    // (top-down). SPIERSedit's GenerateThresh() accesses GA with y increasing
    // downward and does its own vertical flip when compositing. We write top-down
    // here to match the ColArray orientation.

    QByteArray ga(fwidth4 * imageHeight, static_cast<char>(0));

    for (int y = 0; y < imageHeight; y++)
    {
        for (int x = 0; x < imageWidth; x++)
        {
            int srcIdx  = y * imageWidth + x;
            int dstByte = y * fwidth4 + x;
            ga[dstByte] = static_cast<char>(state.alpha[static_cast<size_t>(srcIdx)]);
        }
    }

    return ga;
}
