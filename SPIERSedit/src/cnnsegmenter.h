/**
 * @file
 * Header: CNN Segmenter
 *
 * A compact U-Net segmentation model (LibTorch) plus in-app training and
 * whole-slice inference, for the CNN-based generation method. Trains from the
 * sparse locked-pixel labels gathered by CnnData, aiming to beat the
 * random-forest ML on accuracy by learning multi-scale spatial features.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 */

#ifndef CNNSEGMENTER_H
#define CNNSEGMENTER_H

// torch first (before any Qt header) — Qt keyword macros collide with LibTorch.
#include <torch/torch.h>

#include "cnndata.h"

#include <memory>
#include <vector>

/**
 * @brief Compact U-Net for N-class segmentation of single-channel patches.
 *
 * Two down/up levels with skip connections; small base-channel count so it
 * trains quickly on CPU from a modest number of painted slices. Output is
 * per-pixel class logits [B, numClasses, H, W].
 */
struct UNetImpl : torch::nn::Module
{
    UNetImpl(int inChannels, int numClasses, int baseChannels = 16);
    torch::Tensor forward(torch::Tensor x);

    // encoder
    torch::nn::Sequential enc1{nullptr}, enc2{nullptr};
    // bottleneck
    torch::nn::Sequential bottleneck{nullptr};
    // decoder (upconv + conv block per level)
    torch::nn::ConvTranspose2d up2{nullptr};
    torch::nn::Sequential dec2{nullptr};
    torch::nn::ConvTranspose2d up1{nullptr};
    torch::nn::Sequential dec1{nullptr};
    // head
    torch::nn::Conv2d head{nullptr};

    torch::nn::MaxPool2d pool{nullptr};
};
TORCH_MODULE(UNet);

/**
 * @brief Owns a UNet, trains it from CnnData slices, and runs inference.
 */
class CnnSegmenter
{
public:
    CnnSegmenter();

    /// True once a model has been trained (or loaded).
    bool isTrained() const { return trained; }

    /**
     * @brief Train the U-Net from the given training slices.
     * @param slices      Training slices (image + sparse labels).
     * @param numClasses  Number of segment classes.
     * @param epochs      Training iterations (batches).
     * @param patchSize   Patch side length.
     * @param batchSize   Patches per iteration.
     * @param lastLoss    Out: final loss value (for reporting).
     * @return true on success.
     */
    bool train(const std::vector<CnnSlice> &slices, int numClasses,
               int epochs, int patchSize, int batchSize, double &lastLoss);

    /**
     * @brief Run inference on a full-slice image tensor [1, H, W] (0..1).
     * @return per-pixel class-probability tensor [numClasses, H, W] (softmax).
     */
    torch::Tensor predict(const torch::Tensor &image);

    int classes() const { return numClasses_; }

private:
    UNet model{nullptr};
    torch::Device device{torch::kCPU};
    bool trained = false;
    int numClasses_ = 0;
};

#endif // CNNSEGMENTER_H
