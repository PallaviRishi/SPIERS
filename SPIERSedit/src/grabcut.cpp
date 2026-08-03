/**
 * @file
 * Source: GMM-based Segmentation Engine
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2026 by the SPIERS contributors.
 */

#include "grabcut.h"

#include <QImage>

#include <algorithm>
#include <cmath>

// ─── Constructor ──────────────────────────────────────────────────────────────

GrabCut::GrabCut() {}

// ─── Input setters ────────────────────────────────────────────────────────────

void GrabCut::setImage(const QImage &img)
{
    QImage src = img.convertToFormat(QImage::Format_RGB32);

    imageWidth = src.width();
    imageHeight = src.height();
    int pixelCount = imageWidth * imageHeight;
    pixels.resize(static_cast<size_t>(pixelCount) * 3);

    const uchar *bits = src.bits();
    for (int y = 0; y < imageHeight; y++)
    {
        for (int x = 0; x < imageWidth; x++)
        {
            int i = pixelIndex(x, y);
            int byteOffset = (y * src.bytesPerLine()) + x * 4;
            pixels[static_cast<size_t>(i) * 3 + 0] = static_cast<double>(bits[byteOffset + 2]) / 255.0;
            pixels[static_cast<size_t>(i) * 3 + 1] = static_cast<double>(bits[byteOffset + 1]) / 255.0;
            pixels[static_cast<size_t>(i) * 3 + 2] = static_cast<double>(bits[byteOffset + 0]) / 255.0;
        }
    }
}

void GrabCut::setTrimap(const QByteArray &trimapData)
{
    int pixelCount = imageWidth * imageHeight;
    trimap.resize(static_cast<size_t>(pixelCount));
    for (int i = 0; i < pixelCount; i++)
        trimap[static_cast<size_t>(i)] = static_cast<uchar>(trimapData.at(i));
}

void GrabCut::setState(const GrabCutState &savedState)
{
    state = savedState;
    int pixelCount = imageWidth * imageHeight;
    for (int i = 0; i < pixelCount; i++)
    {
        if (trimap[static_cast<size_t>(i)] == TRIMAP_FOREGROUND)
            state.alpha[static_cast<size_t>(i)] = ALPHA_FG;
        else if (trimap[static_cast<size_t>(i)] == TRIMAP_BACKGROUND)
            state.alpha[static_cast<size_t>(i)] = ALPHA_BG;
    }
    state.isInitialised = true;
}

// ─── Initialisation ───────────────────────────────────────────────────────────

void GrabCut::initAlphaFromTrimap()
{
    int pixelCount = imageWidth * imageHeight;
    state.alpha.resize(static_cast<size_t>(pixelCount));
    state.componentMap.resize(static_cast<size_t>(pixelCount), 0);

    for (int i = 0; i < pixelCount; i++)
    {
        if (trimap[static_cast<size_t>(i)] == TRIMAP_FOREGROUND)
            state.alpha[static_cast<size_t>(i)] = ALPHA_FG;
        else
            state.alpha[static_cast<size_t>(i)] = ALPHA_BG;
    }
}

void GrabCut::initialise()
{
    initAlphaFromTrimap();

    int pixelCount = imageWidth * imageHeight;
    std::vector<double> fgPixels;
    std::vector<double> bgPixels;
    fgPixels.reserve(static_cast<size_t>(pixelCount) * 3);
    bgPixels.reserve(static_cast<size_t>(pixelCount) * 3);

    for (int i = 0; i < pixelCount; i++)
    {
        double r;
        double g;
        double b;
        getPixel(i, r, g, b);
        if (trimap[static_cast<size_t>(i)] == TRIMAP_FOREGROUND)
        {
            fgPixels.push_back(r);
            fgPixels.push_back(g);
            fgPixels.push_back(b);
        }
        else
        {
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

// ─── Classification ───────────────────────────────────────────────────────────

/**
 * @brief Classify each unknown pixel by comparing fg and bg GMM likelihoods.
 *        Pixels with higher P(z|fg) are assigned ALPHA_FG, otherwise ALPHA_BG.
 *        Trimap-forced pixels are always honoured.
 */
void GrabCut::classifyByGmm()
{
    int pixelCount = imageWidth * imageHeight;
    for (int i = 0; i < pixelCount; i++)
    {
        uchar t = trimap[static_cast<size_t>(i)];
        if (t == TRIMAP_FOREGROUND)
            state.alpha[static_cast<size_t>(i)] = ALPHA_FG;
        else if (t == TRIMAP_BACKGROUND)
            state.alpha[static_cast<size_t>(i)] = ALPHA_BG;
        else
        {
            double r;
            double g;
            double b;
            getPixel(i, r, g, b);
            double pFg = state.fgGmm.probability(r, g, b);
            double pBg = state.bgGmm.probability(r, g, b);
            state.alpha[static_cast<size_t>(i)] = (pFg > pBg) ? ALPHA_FG : ALPHA_BG;
        }
    }
}

// ─── Public run interface ─────────────────────────────────────────────────────

void GrabCut::runOneIteration()
{
    if (!state.isInitialised)
        initialise();

    classifyByGmm();
}

void GrabCut::run(int iterations)
{
    if (!state.isInitialised)
        initialise();

    // Classify using the initial GMMs (first pass always uses the scribble-seeded models)
    classifyByGmm();

    if (progressCb)
        progressCb(50);

    // Additional iterations: refit GMMs from current alpha then reclassify.
    // This can improve results if the initial scribbles are sparse.
    for (int iter = 1; iter < iterations; iter++)
    {
        if (progressCb)
            progressCb(static_cast<int>(50 + 50.0 * iter / iterations));

        // Refit GMMs from current classification
        state.fgGmm.resetAccumulators();
        state.bgGmm.resetAccumulators();

        int pixelCount = imageWidth * imageHeight;
        for (int i = 0; i < pixelCount; i++)
        {
            double r;
            double g;
            double b;
            getPixel(i, r, g, b);
            int comp = state.fgGmm.mostLikelyComponent(r, g, b);

            if (state.alpha[static_cast<size_t>(i)] == ALPHA_FG)
                state.fgGmm.addSample(comp, r, g, b);
            else
                state.bgGmm.addSample(state.bgGmm.mostLikelyComponent(r, g, b), r, g, b);
        }

        state.fgGmm.learnFromAccumulated();
        state.bgGmm.learnFromAccumulated();

        // Reclassify with updated GMMs
        classifyByGmm();
    }

    if (progressCb)
        progressCb(100);
}

// ─── Output ───────────────────────────────────────────────────────────────────

QByteArray GrabCut::alphaAsGAImage(int fwidth4) const
{
    QByteArray ga(fwidth4 * imageHeight, static_cast<char>(0));

    for (int y = 0; y < imageHeight; y++)
    {
        for (int x = 0; x < imageWidth; x++)
        {
            int srcIdx = pixelIndex(x, y);
            int dstByte = y * fwidth4 + x;
            ga[dstByte] = static_cast<char>(state.alpha[static_cast<size_t>(srcIdx)]);
        }
    }

    return ga;
}
