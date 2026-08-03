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
 * 1. User fully segments one or more seed slices via the GrabCut dialog.
 * 2. User triggers propagation with a chosen slice range.
 * 3. For each target slice in order from the seed outward:
 *    a. Load the colour image for that slice.
 *    b. Build a trimap (all TRIMAP_UNKNOWN unless the slice has stored scribbles).
 *    c. Warm-start GrabCut with the GMMs from the most recently propagated slice.
 *    d. Run a reduced number of GrabCut iterations (default 3).
 *    e. Write the result into GA[] for that slice and mark it dirty.
 *    f. Update the propagated state so the next slice uses freshly-estimated GMMs.
 * 4. Progress and completion are reported back via Qt signals.
 *
 * Confidence estimation
 * ---------------------
 * Confidence is estimated as the fraction of UNKNOWN pixels that did not
 * change assignment between two consecutive iterations. Slices below the
 * confidence threshold are flagged for manual review.
 *
 * Thread safety
 * -------------
 * propagate() is designed to be called from a background QThread.
 * Progress and completion are signalled back to the main thread via Qt signals.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2026 by the SPIERS contributors.
 */

#ifndef SLICEPROPAGATION_H
#define SLICEPROPAGATION_H

#include "grabcut.h"

#include <QByteArray>
#include <QImage>
#include <QMetaType>
#include <QObject>
#include <QString>
#include <QStringList>

#include <vector>

/**
 * @brief Result record for one propagated slice.
 */
struct PropagationResult
{
    double confidence;  ///< 0-1, fraction of pixels that stabilised (higher = better)
    bool needsReview;   ///< True if confidence < confidenceThreshold
    int sliceIndex;     ///< Index into the Files[] list
    bool success;       ///< Whether the slice was processed without error
};

/**
 * @brief Parameters controlling a propagation pass.
 */
struct PropagationParams
{
    double confidenceThreshold; ///< Slices below this value are flagged for review
    int firstSlice;             ///< First slice to propagate to (inclusive)
    int iterations;             ///< GrabCut iterations per slice
    int lastSlice;              ///< Last slice to propagate to (inclusive)
    bool overwriteExisting;     ///< If false, slices with existing GA data are skipped
    int seedSlice;              ///< Slice index of the already-segmented seed
    int segmentIndex;           ///< Which segment index (GA[seg]) to write results into

    PropagationParams()
        : confidenceThreshold(0.95)
        , firstSlice(0)
        , iterations(3)
        , lastSlice(0)
        , overwriteExisting(true)
        , seedSlice(0)
        , segmentIndex(0)
    {}
};

/**
 * @brief Manages inter-slice GrabCut propagation across the tomogram stack.
 *
 * Emits Qt signals for progress and completion. Connect these to the main
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
     * @param seedState  GrabCut state to propagate from.
     */
    void setSeedState(const GrabCutState &seedState);

    /**
     * @brief Set the per-slice trimap data.
     *        trimaps[i] is the trimap for slice i; may be empty (all UNKNOWN).
     *        The size must equal the total number of slices in the dataset.
     * @param trimaps  Per-slice trimap byte arrays.
     */
    void setTrimaps(const std::vector<QByteArray> &trimaps);

    /**
     * @brief Set the source colour image file list (mirrors SPIERSedit Files[]).
     * @param files  List of absolute file paths.
     */
    void setFileList(const QStringList &files);

    /**
     * @brief Set the image dimensions for all slices in the dataset.
     * @param imageWidth   Pixel width.
     * @param imageHeight  Pixel height.
     * @param fw4          Padded row stride (fwidth4) used by SPIERSedit GA[] images.
     */
    void setImageDimensions(int imageWidth, int imageHeight, int fw4);

    /**
     * @brief Set propagation parameters.
     * @param params  Parameters to use.
     */
    void setParams(const PropagationParams &params);

    /**
     * @brief Cancel a running propagation at the next slice boundary.
     */
    void cancel();

    /**
     * @brief Run the propagation. Designed to be called from a background QThread.
     *        Emits progressUpdated() and propagationComplete() during execution.
     * @return  Per-slice results (also delivered via propagationComplete() signal).
     */
    std::vector<PropagationResult> propagate();

    /**
     * @brief Returns true if any slice in the last run was flagged for review.
     * @return  True if at least one result has needsReview set.
     */
    bool hasLowConfidenceSlices() const;

    /**
     * @brief Returns the slice indices flagged as needing manual review.
     * @return  Vector of slice indices.
     */
    std::vector<int> lowConfidenceSlices() const;

signals:
    /**
     * @brief Emitted periodically during propagation.
     * @param percent   Overall progress 0-100.
     * @param sliceIdx  Index of the slice currently being processed.
     */
    void progressUpdated(int percent, int sliceIdx);

    /**
     * @brief Emitted when propagation finishes or is cancelled.
     * @param results    Per-slice results.
     * @param cancelled  True if the run was stopped early by cancel().
     */
    void propagationComplete(std::vector<PropagationResult> results, bool cancelled);

private:
    bool cancelFlag;                     ///< Set by cancel() to stop the propagation loop
    int fwidth4;                         ///< Padded row stride for GA[] images
    QStringList files;                   ///< Source image file paths
    int sliceHeight;                     ///< Image height in pixels
    int sliceWidth;                      ///< Image width in pixels
    GrabCutState seedState;              ///< Seed GMM state from the annotated slice
    PropagationParams params;            ///< Active propagation parameters
    std::vector<PropagationResult> results; ///< Accumulated per-slice results
    std::vector<QByteArray> trimaps;     ///< Per-slice trimap data

    /**
     * @brief Estimate algorithmic confidence for the current slice by running
     *        one extra iteration and measuring how many pixels changed assignment.
     * @param gc      GrabCut engine (modified in place by the extra iteration).
     * @param trimap  Trimap for this slice (used to identify UNKNOWN pixels).
     * @return  Confidence value in range 0-1.
     */
    double estimateConfidence(GrabCut &gc, const QByteArray &trimap) const;

    /**
     * @brief Load and decode the colour image for the given slice index.
     * @param sliceIndex  Index into the files list.
     * @return  Loaded QImage, or a null QImage on failure.
     */
    QImage loadSliceImage(int sliceIndex) const;

    /**
     * @brief Propagate through a contiguous range of slices in one direction.
     * @param from         Starting slice (exclusive — this slice is already done).
     * @param to           End slice (inclusive).
     * @param step         +1 for forward, -1 for backward.
     * @param workingState Mutable GrabCut state; updated after each slice.
     * @param totalSlices  Total slices being processed (for progress percentage).
     * @param doneCount    Running count of completed slices (updated in place).
     */
    void propagateRange(int from, int to, int step,
                        GrabCutState &workingState,
                        int totalSlices, int &doneCount);

    /**
     * @brief Return the stored trimap for slice i, or an all-UNKNOWN trimap
     *        if no scribbles have been recorded for that slice.
     * @param sliceIndex  Slice index.
     * @return  Trimap byte array (size = sliceWidth * sliceHeight).
     */
    QByteArray trimapForSlice(int sliceIndex) const;

    /**
     * @brief Write the GrabCut alpha result into the in-memory GA[] image for
     *        the given slice and mark it dirty for saving.
     * @param sliceIndex  Slice index.
     * @param gaData      Alpha data in SPIERSedit GA[] stride format.
     */
    void writeResultToGA(int sliceIndex, const QByteArray &gaData) const;
};

Q_DECLARE_METATYPE(PropagationResult)
Q_DECLARE_METATYPE(std::vector<PropagationResult>)

#endif // SLICEPROPAGATION_H
