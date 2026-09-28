/**
 * @file
 * Header: CNN Interface
 *
 * Orchestrates the CNN segmentation generation method: gather training data
 * from locked pixels, train the U-Net, then classify selected slices and write
 * the per-segment results into the GA[] working images — mirroring
 * MLInterface::Generate so it behaves as a peer generation method.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 */

#ifndef CNNINTERFACE_H
#define CNNINTERFACE_H

// torch first (before any Qt header).
#include "cnnsegmenter.h"

#include <memory>

class QListWidget;

class CnnInterface
{
public:
    CnnInterface();

    /**
     * @brief Full pipeline: train from locked-pixel samples across the selected
     *        slices, then classify every selected slice and write results into
     *        the segment working images (GA[]), honouring mask/segment locking.
     *
     * @param sliceSelector  The slice selector widget.
     * @param samplePercent  Sub-sampling percentage for training labels.
     * @param epochs         Training iterations.
     * @param patchSize      Patch side length.
     * @param batchSize      Patches per iteration.
     */
    void Generate(QListWidget *sliceSelector,
                  int samplePercent = 100,
                  int epochs = 300,
                  int patchSize = 64,
                  int batchSize = 16);

private:
    std::unique_ptr<CnnSegmenter> segmenter;
};

#endif // CNNINTERFACE_H
