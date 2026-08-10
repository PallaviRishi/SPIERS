/**
 * @file
 * Header: GMM Auto-Sample Generator
 *
 * Provides automatic training sample generation for the ML (Random Forest)
 * pipeline using a Gaussian Mixture Model. Instead of the user manually
 * painting thousands of training pixels, this module:
 *
 *   1. Takes a small set of user scribbles (a few strokes per class)
 *   2. Fits a 5-component GMM to each class's colour distribution
 *   3. Classifies all pixels by GMM likelihood
 *   4. Exports high-confidence pixels as locked+labelled training points
 *      in the format expected by MLInterface::Sample()
 *
 * This bridges the interactive GMM segmentation with the Random Forest
 * system: GMM provides an instant first-pass classification that seeds
 * the RF training sample, eliminating the need for tedious manual sample
 * painting across multiple slices.
 *
 * Usage from MLInterface:
 * @code
 *   #include "gmmautosample.h"
 *   QVector<LabelledPoint> autoLabels = gmmAutoSample(
 *       CurrentFile,           // current slice index
 *       confidenceThreshold    // e.g. 0.8 — only export pixels with strong GMM confidence
 *   );
 *   labels.append(autoLabels); // add to ML training set
 * @endcode
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2026 by the SPIERS contributors.
 */

#ifndef GMMAUTOSAMPLE_H
#define GMMAUTOSAMPLE_H

#include <QByteArray>
#include <QImage>
#include <QVector>

/**
 * @brief A labelled training point for the ML system.
 *        Compatible with the LabelledPoint class in Mark's ML code.
 */
struct GmmLabelledPoint
{
    int x;        ///< X pixel coordinate
    int y;        ///< Y pixel coordinate
    int z;        ///< Slice index
    int segment;  ///< Segment index (0-based)
};

/**
 * @brief Run GMM classification on a single slice and return high-confidence
 *        pixels as labelled training points for the Random Forest.
 *
 * The function reads the colour image (ColArray) and the current trimap
 * (derived from locked pixels in the current segment assignments), fits
 * GMMs, classifies pixels, and returns those where the GMM confidence
 * ratio exceeds the threshold.
 *
 * @param sliceIndex           The slice to process.
 * @param colourImage          The source colour image for this slice.
 * @param trimap               One byte per pixel: 255=FG scribble, 0=BG scribble, 128=unknown.
 * @param fgSegmentIndex       Which segment index the foreground class maps to.
 * @param bgSegmentIndex       Which segment index the background class maps to.
 * @param confidenceThreshold  Minimum likelihood ratio (pWinner/pLoser) to include
 *                             a pixel in the output. Higher = fewer but more reliable points.
 *                             Typical values: 2.0–5.0.
 * @param maxPointsPerClass    Maximum number of points to return per class (to avoid
 *                             overwhelming the RF with too-large samples). 0 = no limit.
 * @return  Vector of labelled points suitable for appending to the ML training set.
 */
QVector<GmmLabelledPoint> gmmAutoSample(
    int sliceIndex,
    const QImage &colourImage,
    const QByteArray &trimap,
    int fgSegmentIndex,
    int bgSegmentIndex,
    double confidenceThreshold = 3.0,
    int maxPointsPerClass = 20000
);

/**
 * @brief Convenience function: generate a trimap from the current locks and
 *        segment assignments on a given slice.
 *
 * Reads the Locks[] and GA[] arrays for the specified slice and builds a
 * trimap where locked pixels assigned to fgSegment become TRIMAP_FOREGROUND
 * and locked pixels assigned to bgSegment become TRIMAP_BACKGROUND.
 *
 * @param sliceIndex       Slice to read from.
 * @param fgSegmentIndex   Segment index for foreground.
 * @param bgSegmentIndex   Segment index for background.
 * @return  Trimap byte array (size = fwidth * fheight).
 */
QByteArray buildTrimapFromLocks(int sliceIndex, int fgSegmentIndex, int bgSegmentIndex);

#endif // GMMAUTOSAMPLE_H
