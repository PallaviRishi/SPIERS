/**
 * @file
 * Header: Inter-slice Propagation
 *
 * Propagates a GrabCut segmentation from a seeded slice to adjacent slices
 * through the tomogram stack. The fitted foreground and background GMMs from
 * a segmented slice are used as warm-start colour models for neighbouring
 * slices, so users only need to draw scribbles on a small number of slices
 * rather than every one.
 *
 * Strategy
 * --------
 * 1. User fully segments one or more "seed" slices (via the GrabCut dialog).
 * 2. User triggers "Propagate to adjacent slices" with a chosen range.
 * 3. For each target slice in order from the seed outward:
 *    a. Load the colour image for that slice.
 *    b. Build an empty trimap (all TRIMAP_UNKNOWN — no new scribbles).
 *       If the target slice already has any user scribbles, those are honoured.
 *    c. Warm-start GrabCut with the GMMs from the most recently propagated
 *       neighbour (the GMMs drift slowly as the fossil changes across slices).
 *    d. Run a reduced number of GrabCut iterations (default 3).
 *    e. Write the result into GA[] for that slice and save to disk.
 *    f. Update the propagated state so the next slice can use the freshly
 *       re-estimated GMMs rather than the original seed's GMMs.
 * 4. A progress callback is used to update the UI.
 *
 * Confidence decay
 * ----------------
 * As propagation moves further from the seed, colour models may drift and
 * the segmentation quality degrades. We track a "confidence" value per slice
 * that decays with distance from the seed. Slices below a confidence threshold
 * are flagged so the user knows they need manual checking. The confidence is
 * estimated as the fraction of pixels that changed assignment between
 * consecutive iterations (low change = high confidence).
 *
 * Thread safety
 * -------------
 * propagate() is designed to be called from SPIERSedit's BackThread (QThread)
 * with progress and completion reported back to the main thread via Qt signals.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2024 by the SPIERS contributors.
 */

#ifndef SLICEPROPAGATION_H
#define SLICEPROPAGATION_H

#include <QObject>
#include <QImage>
#include <QByteArray>
#include <QString>
#include <QStringList>
#include <QMetaType>
#include <vector>
#include <functional>
#include "grabcut.h"

/**
 * @brief Result record for one propagated slice.
 */
struct PropagationResult
{
    int   sliceIndex;     ///< Index into the Files[] list
    bool  success;        ///< Whether the slice was processed without error
    double confidence;    ///< 0–1, fraction of pixels that stabilised (higher = better)
    bool  needsReview;    ///< True if confidence < confidenceThreshold
};

/**
 * @brief Parameters controlling the propagation pass.
 */
struct PropagationParams
{
    int seedSlice;         ///< Slice index of the already-segmented seed
    int firstSlice;        ///< First slice to propagate to (inclusive)
    int lastSlice;         ///< Last slice to propagate to (inclusive)
    int segmentIndex;      ///< Which segment (GA[seg]) to write results into
    int iterations;        ///< GrabCut iterations per slice (default 3)
    double confidenceThreshold; ///< Below this → needsReview=true (default 0.95)
    bool overwriteExisting;     ///< If false, skip slices that already have non-trivial GA data

    PropagationParams()
        : seedSlice(0), firstSlice(0), lastSlice(0), segmentIndex(0),
          iterations(3), confidenceThreshold(0.95), overwriteExisting(true) {}
};

/**
 * @brief Manages inter-slice GrabCut propagation across the tomogram stack.
 *
 * Emits Qt signals for progress and completion — connect these to the main
 * window before calling propagate() on a background thread.
 */
class SlicePropagation : public QObject
{
    Q_OBJECT

public:
    explicit SlicePropagation(QObject *parent = nullptr);

    /**
     * @brief Set the seed state (GMMs + alpha) from the already-segmented slice.
     *        Must be called before propagate().
     */
    void setSeedState(const GrabCutState &state);

    /**
     * @brief Set the per-slice trimap data for forced scribble regions.
     *        trimaps[i] is the trimap for slice i. May be empty (all UNKNOWN).
     *        Size must equal the total number of slices in the dataset.
     */
    void setTrimaps(const std::vector<QByteArray> &trimaps);

    /**
     * @brief Set the source colour image file list (mirrors SPIERSedit Files[]).
     */
    void setFileList(const QStringList &files);

    /**
     * @brief Image dimensions — must match all slice images.
     */
    void setImageDimensions(int width, int height, int fwidth4);

    /**
     * @brief Set propagation parameters.
     */
    void setParams(const PropagationParams &params);

    /**
     * @brief Cancel a running propagation at the next slice boundary.
     *        Thread-safe (sets an atomic flag).
     */
    void cancel();

    /**
     * @brief Run the propagation. Designed to be called from a background
     *        QThread. Emits progressUpdated() and propagationComplete() signals.
     *        Returns the per-slice results (also available via the signal).
     */
    std::vector<PropagationResult> propagate();

    /**
     * @brief Returns true if any slices in the last run were flagged for review.
     */
    bool hasLowConfidenceSlices() const;

    /**
     * @brief Returns slice indices flagged as needing manual review.
     */
    std::vector<int> lowConfidenceSlices() const;

signals:
    /**
     * @brief Emitted periodically during propagation.
     * @param percent   Overall progress 0–100
     * @param sliceIdx  Current slice index being processed
     */
    void progressUpdated(int percent, int sliceIdx);

    /**
     * @brief Emitted when propagation finishes (or is cancelled).
     * @param results   Per-slice results
     * @param cancelled True if stopped early by cancel()
     */
    void propagationComplete(std::vector<PropagationResult> results, bool cancelled);

private:
    GrabCutState   seedState_;
    std::vector<QByteArray> trimaps_;
    QStringList    files_;
    PropagationParams params_;
    int W_ = 0, H_ = 0, fwidth4_ = 0;

    std::vector<PropagationResult> results_;
    volatile bool cancelFlag_ = false;

    /**
     * @brief Load and decode one colour slice from disk.
     *        Returns a null QImage on failure.
     */
    QImage loadSliceImage(int sliceIndex) const;

    /**
     * @brief Load the existing trimap for slice i, or return an all-UNKNOWN
     *        trimap if none is stored.
     */
    QByteArray trimapForSlice(int sliceIndex) const;

    /**
     * @brief Write the resulting GA image data directly into SPIERSedit's
     *        in-memory GA[] array and mark the slice dirty for saving.
     */
    void writeResultToGA(int sliceIndex, const QByteArray &gaData) const;

    /**
     * @brief Estimate confidence for a slice: fraction of UNKNOWN pixels
     *        that did not change assignment between the last two iterations.
     *        Implemented by running one extra iteration and comparing.
     */
    double estimateConfidence(GrabCut &gc, const QByteArray &trimap) const;

    /**
     * @brief Propagate one direction (forward or backward from seed).
     * @param from   Start slice (exclusive — this slice is already done)
     * @param to     End slice (inclusive)
     * @param step   +1 for forward, -1 for backward
     * @param state  Mutable working state, updated after each slice
     * @param totalSlices  Total number of slices being processed (for progress)
     * @param doneCount    Reference to progress counter
     */
    void propagateRange(int from, int to, int step,
                        GrabCutState &state,
                        int totalSlices, int &doneCount);
};

#endif // SLICEPROPAGATION_H

Q_DECLARE_METATYPE(PropagationResult)
Q_DECLARE_METATYPE(std::vector<PropagationResult>)
