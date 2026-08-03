/**
 * @file
 * Source: Inter-slice Propagation
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2024 by the SPIERS contributors.
 */

#include "slicepropagation.h"
#include "fileio.h"
#include "globals.h"

#include <QFile>
#include <QFileInfo>
#include <QDebug>
#include <cmath>
#include <algorithm>

// Register metatypes once at startup for cross-thread signal/slot delivery
static bool metaTypesRegistered = []() {
    qRegisterMetaType<PropagationResult>("PropagationResult");
    qRegisterMetaType<std::vector<PropagationResult>>("std::vector<PropagationResult>");
    return true;
}();

// ─── Constructor ──────────────────────────────────────────────────────────────

SlicePropagation::SlicePropagation(QObject *parent)
    : QObject(parent)
    , cancelFlag_(false)
{}

// ─── Setup ────────────────────────────────────────────────────────────────────

void SlicePropagation::setSeedState(const GrabCutState &state)
{
    seedState_ = state;
}

void SlicePropagation::setTrimaps(const std::vector<QByteArray> &trimaps)
{
    trimaps_ = trimaps;
}

void SlicePropagation::setFileList(const QStringList &files)
{
    files_ = files;
}

void SlicePropagation::setImageDimensions(int width, int height, int fw4)
{
    W_       = width;
    H_       = height;
    fwidth4_ = fw4;
}

void SlicePropagation::setParams(const PropagationParams &params)
{
    params_ = params;
}

void SlicePropagation::cancel()
{
    cancelFlag_ = true;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

QImage SlicePropagation::loadSliceImage(int sliceIndex) const
{
    if (sliceIndex < 0 || sliceIndex >= files_.size())
        return QImage();

    QImage img(files_.at(sliceIndex));
    if (img.isNull())
    {
        qWarning() << "SlicePropagation: could not load image for slice" << sliceIndex
                   << ":" << files_.at(sliceIndex);
    }
    return img;
}

QByteArray SlicePropagation::trimapForSlice(int sliceIndex) const
{
    if (sliceIndex >= 0 && sliceIndex < static_cast<int>(trimaps_.size()))
    {
        const QByteArray &tm = trimaps_.at(static_cast<size_t>(sliceIndex));
        if (!tm.isEmpty())
            return tm;
    }
    // No stored trimap — return all-UNKNOWN
    return QByteArray(W_ * H_, static_cast<char>(TRIMAP_UNKNOWN));
}

void SlicePropagation::writeResultToGA(int sliceIndex, const QByteArray &gaData) const
{
    // GA[seg] is a QImage (Format_Indexed8) with stride fwidth4_.
    // We set the pixel data directly then mark the slice dirty so fileio
    // will save it on the next autosave / explicit save.
    if (params_.segmentIndex < 0 || params_.segmentIndex >= SegmentCount)
        return;

    QImage *gaImage = GA.at(params_.segmentIndex);
    if (!gaImage || gaImage->isNull())
        return;

    uchar *bits = gaImage->bits();
    const int len = fwidth4_ * H_;
    for (int b = 0; b < len && b < gaData.size(); b++)
        bits[b] = static_cast<uchar>(gaData.at(b));

    // Mark dirty so the slice gets written to disk
    if (sliceIndex >= 0 && sliceIndex < FilesDirty.size())
        FilesDirty[sliceIndex] = true;

    if (params_.segmentIndex < Segments.size())
        Segments[params_.segmentIndex]->Dirty = true;
}

// ─── Confidence estimation ────────────────────────────────────────────────────

double SlicePropagation::estimateConfidence(GrabCut &gc, const QByteArray &trimap) const
{
    // Snapshot alpha before an extra iteration
    const std::vector<uchar> alphaBefore = gc.alpha();

    // Run one more iteration
    gc.runOneIteration();

    const std::vector<uchar> &alphaAfter = gc.alpha();

    // Count UNKNOWN pixels that changed
    int unknownCount  = 0;
    int changedCount  = 0;
    int N = W_ * H_;

    for (int i = 0; i < N; i++)
    {
        if (static_cast<uchar>(trimap.at(i)) == TRIMAP_UNKNOWN)
        {
            unknownCount++;
            if (alphaBefore[static_cast<size_t>(i)] != alphaAfter[static_cast<size_t>(i)])
                changedCount++;
        }
    }

    if (unknownCount == 0) return 1.0;
    return 1.0 - static_cast<double>(changedCount) / static_cast<double>(unknownCount);
}

// ─── Core propagation range ───────────────────────────────────────────────────

/**
 * @brief Propagate in one direction through the stack.
 *
 * For each slice from 'from+step' to 'to' (step = ±1):
 *   1. Load image
 *   2. Build trimap (user scribbles if any, else all-UNKNOWN)
 *   3. Check overwrite policy
 *   4. Warm-start GrabCut from previous slice's state
 *   5. Run iterations
 *   6. Estimate confidence
 *   7. Write result to GA[]
 *   8. Update state for next iteration (GMMs drift with the slice)
 */
void SlicePropagation::propagateRange(int from, int to, int step,
                                       GrabCutState &state,
                                       int totalSlices, int &doneCount)
{
    int current = from + step;
    while ((step > 0 ? current <= to : current >= to) && !cancelFlag_)
    {
        PropagationResult res;
        res.sliceIndex  = current;
        res.success     = false;
        res.confidence  = 0.0;
        res.needsReview = true;

        // ── 1. Load image ─────────────────────────────────────────────────────
        QImage img = loadSliceImage(current);
        if (img.isNull())
        {
            qWarning() << "SlicePropagation: skipping slice" << current << "(load failed)";
            results_.push_back(res);
            current += step;
            doneCount++;
            int pct = static_cast<int>(100.0 * doneCount / totalSlices);
            emit progressUpdated(pct, current);
            continue;
        }

        // ── 2. Trimap ─────────────────────────────────────────────────────────
        QByteArray tm = trimapForSlice(current);

        // ── 3. Overwrite check ────────────────────────────────────────────────
        if (!params_.overwriteExisting && params_.segmentIndex < SegmentCount)
        {
            // Check if GA already has non-trivial data: any pixel != 0
            QImage *gaImg = GA.at(params_.segmentIndex);
            if (gaImg && !gaImg->isNull())
            {
                const uchar *bits = gaImg->constBits();
                bool hasData = false;
                for (int b = 0; b < fwidth4_ * H_; b++)
                {
                    if (bits[b] > 0) { hasData = true; break; }
                }
                if (hasData)
                {
                    // Skip — preserve existing manual work
                    res.success     = true;
                    res.confidence  = 1.0;
                    res.needsReview = false;
                    results_.push_back(res);
                    current += step;
                    doneCount++;
                    emit progressUpdated(
                        static_cast<int>(100.0 * doneCount / totalSlices), current);
                    continue;
                }
            }
        }

        // ── 4 & 5. GrabCut warm-start + run ───────────────────────────────────
        GrabCut gc;
        gc.setImage(img);
        gc.setTrimap(tm);
        gc.setState(state);              // warm-start from prev slice

        gc.run(params_.iterations);

        // ── 6. Confidence ─────────────────────────────────────────────────────
        double conf = estimateConfidence(gc, tm);
        res.confidence  = conf;
        res.needsReview = conf < params_.confidenceThreshold;

        // ── 7. Write to GA[] ──────────────────────────────────────────────────
        QByteArray gaData = gc.alphaAsGAImage(fwidth4_);
        writeResultToGA(current, gaData);
        res.success = true;

        // ── 8. Update state for next slice ────────────────────────────────────
        state = gc.getState();

        results_.push_back(res);

        doneCount++;
        int pct = static_cast<int>(100.0 * doneCount / totalSlices);
        emit progressUpdated(pct, current);

        current += step;
    }
}

// ─── Public propagate() ───────────────────────────────────────────────────────

std::vector<PropagationResult> SlicePropagation::propagate()
{
    results_.clear();
    cancelFlag_ = false;

    const int seed  = params_.seedSlice;
    const int first = params_.firstSlice;
    const int last  = params_.lastSlice;

    // Total slices to process (excluding the seed itself)
    int totalSlices = 0;
    if (first <= seed) totalSlices += seed - first;       // slices before seed
    if (last  >= seed) totalSlices += last  - seed;       // slices after seed
    if (totalSlices == 0)
    {
        emit propagationComplete(results_, false);
        return results_;
    }

    int doneCount = 0;

    // We propagate in two passes: forward (seed → last) then backward (seed → first).
    // Each pass starts with a fresh copy of the seed state so the forward and
    // backward GMM drifts are independent.

    // ── Forward: seed+1 → last ────────────────────────────────────────────────
    if (last > seed && !cancelFlag_)
    {
        GrabCutState fwdState = seedState_;
        propagateRange(seed, last, +1, fwdState, totalSlices, doneCount);
    }

    // ── Backward: seed-1 → first ─────────────────────────────────────────────
    if (first < seed && !cancelFlag_)
    {
        GrabCutState bwdState = seedState_;
        propagateRange(seed, first, -1, bwdState, totalSlices, doneCount);
    }

    bool wasCancelled = cancelFlag_;
    emit propagationComplete(results_, wasCancelled);
    return results_;
}

// ─── Result queries ───────────────────────────────────────────────────────────

bool SlicePropagation::hasLowConfidenceSlices() const
{
    for (const PropagationResult &r : results_)
        if (r.needsReview) return true;
    return false;
}

std::vector<int> SlicePropagation::lowConfidenceSlices() const
{
    std::vector<int> out;
    for (const PropagationResult &r : results_)
        if (r.needsReview) out.push_back(r.sliceIndex);
    return out;
}
