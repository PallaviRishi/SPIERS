/**
 * @file
 * Header: CNN Data Pipeline
 *
 * Builds LibTorch training tensors from SPIERSedit's locked-pixel training
 * samples, for the CNN-based segmentation generation method.
 *
 * The supervision signal is sparse: only pixels the user has locked and
 * assigned to a segment carry a label. We therefore produce, per selected
 * training slice:
 *   - an input image tensor  [1, H, W]  (source greyscale, normalised 0..1)
 *   - a label tensor         [H, W]     (segment index at labelled pixels,
 *                                        kIgnoreIndex elsewhere)
 *
 * Training then samples fixed-size patches from these, with augmentation, so a
 * handful of painted slices yields many training examples.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 */

#ifndef CNNDATA_H
#define CNNDATA_H

// NOTE: torch headers MUST be included before any Qt header (Qt #defines
// keywords such as slots/signals/emit that collide with LibTorch identifiers).
#include <torch/torch.h>

#include <vector>

class QListWidget;

/**
 * @brief One training slice: full-slice input image and its sparse label map.
 */
struct CnnSlice
{
    torch::Tensor image;  ///< [1, H, W] float, source greyscale normalised 0..1
    torch::Tensor labels; ///< [H, W] long, segment index or kIgnoreIndex
    int sliceIndex = -1;  ///< originating dataset slice (for reference)
    long labelledPixels = 0; ///< count of non-ignore pixels (training signal)
};

namespace CnnData
{
/// Label value marking a pixel that carries no supervision (excluded from loss).
constexpr long kIgnoreIndex = -100; // matches torch cross-entropy default ignore_index

/// Reseed the patch-sampler RNG so a training run is reproducible.
void ResetSampler(unsigned int seed);

/**
 * @brief Gather full-slice training tensors from all selected slices.
 *
 * Reuses the existing locked-pixel labelling (a locked pixel's class is the
 * segment with the highest GA value, >=128). Unlabelled pixels get kIgnoreIndex.
 *
 * @param sliceSelector  The slice selector widget (selected slices are used).
 * @param percentage     Sub-sampling percentage of locked pixels (as ML uses).
 * @param out            Output list of per-slice tensors (only slices with
 *                       at least one labelled pixel are included).
 * @return total number of labelled (supervised) pixels gathered.
 */
long GatherTrainingSlices(QListWidget *sliceSelector, int percentage,
                          std::vector<CnnSlice> &out);

/**
 * @brief Extract a batch of random augmented patches from the training slices.
 *
 * @param slices     Training slices from GatherTrainingSlices.
 * @param patchSize  Square patch side length (e.g. 64).
 * @param batchSize  Number of patches to return.
 * @param images     Output [B, 1, patchSize, patchSize] float tensor.
 * @param labels     Output [B, patchSize, patchSize] long tensor.
 * @return true if a batch was produced.
 */
bool SamplePatchBatch(const std::vector<CnnSlice> &slices, int patchSize,
                      int batchSize, torch::Tensor &images, torch::Tensor &labels);
}

#endif // CNNDATA_H
