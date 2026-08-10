/**
 * @file
 * Source: GMM Auto-Sample Generator
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2026 by the SPIERS contributors.
 */

#include "gmmautosample.h"
#include "grabcut.h"
#include "globals.h"
#include "fileio.h"

#include <algorithm>
#include <cmath>
#include <random>

QVector<GmmLabelledPoint> gmmAutoSample(
    int sliceIndex,
    const QImage &colourImage,
    const QByteArray &trimap,
    int fgSegmentIndex,
    int bgSegmentIndex,
    double confidenceThreshold,
    int maxPointsPerClass)
{
    QVector<GmmLabelledPoint> result;

    if (colourImage.isNull() || trimap.isEmpty())
        return result;

    // Run GMM segmentation
    GrabCut gc;
    gc.setImage(colourImage);
    gc.setTrimap(trimap);
    gc.initialise();
    gc.run(1);

    // Get the fitted GMMs for confidence scoring
    GrabCutState state = gc.getState();
    int width = gc.width();
    int height = gc.height();

    // Convert image for pixel access
    QImage src = colourImage.convertToFormat(QImage::Format_RGB32);
    const uchar *bits = src.bits();
    int bpl = src.bytesPerLine();

    // Collect high-confidence points
    QVector<GmmLabelledPoint> fgPoints;
    QVector<GmmLabelledPoint> bgPoints;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;
            uchar t = static_cast<uchar>(trimap.at(idx));

            // Skip scribbled pixels — they're already known
            if (t == TRIMAP_FOREGROUND || t == TRIMAP_BACKGROUND)
                continue;

            // Get pixel colour
            int byteOffset = y * bpl + x * 4;
            double r = bits[byteOffset + 2] / 255.0;
            double g = bits[byteOffset + 1] / 255.0;
            double b = bits[byteOffset + 0] / 255.0;

            double pFg = state.fgGmm.probability(r, g, b);
            double pBg = state.bgGmm.probability(r, g, b);

            // Compute confidence ratio
            if (pFg > pBg * confidenceThreshold)
            {
                GmmLabelledPoint pt;
                pt.x = x;
                pt.y = y;
                pt.z = sliceIndex;
                pt.segment = fgSegmentIndex;
                fgPoints.append(pt);
            }
            else if (pBg > pFg * confidenceThreshold)
            {
                GmmLabelledPoint pt;
                pt.x = x;
                pt.y = y;
                pt.z = sliceIndex;
                pt.segment = bgSegmentIndex;
                bgPoints.append(pt);
            }
            // Pixels where neither class dominates are excluded
        }
    }

    // Subsample if we exceed maxPointsPerClass
    std::mt19937 rng(42);

    if (maxPointsPerClass > 0 && fgPoints.size() > maxPointsPerClass)
    {
        std::shuffle(fgPoints.begin(), fgPoints.end(), rng);
        fgPoints.resize(maxPointsPerClass);
    }

    if (maxPointsPerClass > 0 && bgPoints.size() > maxPointsPerClass)
    {
        std::shuffle(bgPoints.begin(), bgPoints.end(), rng);
        bgPoints.resize(maxPointsPerClass);
    }

    result.append(fgPoints);
    result.append(bgPoints);

    return result;
}

QByteArray buildTrimapFromLocks(int sliceIndex, int fgSegmentIndex, int bgSegmentIndex)
{
    QByteArray trimap(fwidth * fheight, static_cast<char>(128));

    // Load locks and segment data for the slice
    LoadAllData(sliceIndex);

    for (int y = 0; y < fheight; y++)
    {
        for (int x = 0; x < fwidth; x++)
        {
            int pos = y * fwidth + x;
            int hpos = (fheight - 1 - y) * fwidth + x;

            // Check if pixel is locked
            if (Locks[hpos * 2])
            {
                // Determine which segment this pixel belongs to (highest GA value)
                int bestSeg = -1;
                int bestVal = 0;
                for (int s = 0; s < SegmentCount; s++)
                {
                    if (!Segments[s]->Activated) continue;
                    int val = static_cast<int>(*(GA[s]->bits() + y * fwidth4 + x));
                    if (val > bestVal)
                    {
                        bestVal = val;
                        bestSeg = s;
                    }
                }

                if (bestSeg == fgSegmentIndex && bestVal >= 128)
                    trimap[pos] = static_cast<char>(255);
                else if (bestSeg == bgSegmentIndex || bestVal < 128)
                    trimap[pos] = static_cast<char>(0);
            }
        }
    }

    return trimap;
}
