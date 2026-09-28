/**
 * @file
 * Source: CNN Segmenter
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 */

// torch first (before any Qt header).
#include "cnnsegmenter.h"

#include <QDebug>

// ---- helper: a (conv -> BN -> ReLU) x2 block ----
static torch::nn::Sequential convBlock(int inC, int outC)
{
    return torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(inC, outC, 3).padding(1)),
        torch::nn::BatchNorm2d(outC),
        torch::nn::ReLU(torch::nn::ReLUOptions(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(outC, outC, 3).padding(1)),
        torch::nn::BatchNorm2d(outC),
        torch::nn::ReLU(torch::nn::ReLUOptions(true)));
}

UNetImpl::UNetImpl(int inChannels, int numClasses, int baseChannels)
{
    const int c1 = baseChannels;      // level 1
    const int c2 = baseChannels * 2;  // level 2
    const int cb = baseChannels * 4;  // bottleneck

    enc1 = register_module("enc1", convBlock(inChannels, c1));
    enc2 = register_module("enc2", convBlock(c1, c2));
    bottleneck = register_module("bottleneck", convBlock(c2, cb));

    up2 = register_module("up2", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(cb, c2, 2).stride(2)));
    dec2 = register_module("dec2", convBlock(c2 + c2, c2));

    up1 = register_module("up1", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(c2, c1, 2).stride(2)));
    dec1 = register_module("dec1", convBlock(c1 + c1, c1));

    head = register_module("head", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(c1, numClasses, 1)));

    pool = register_module("pool", torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(2).stride(2)));
}

torch::Tensor UNetImpl::forward(torch::Tensor x)
{
    // Encoder
    torch::Tensor e1 = enc1->forward(x);        // [B, c1, H,   W]
    torch::Tensor e2 = enc2->forward(pool->forward(e1)); // [B, c2, H/2, W/2]
    torch::Tensor b  = bottleneck->forward(pool->forward(e2)); // [B, cb, H/4, W/4]

    // Decoder with skip connections
    torch::Tensor d2 = up2->forward(b);         // [B, c2, H/2, W/2]
    d2 = torch::cat({d2, e2}, 1);
    d2 = dec2->forward(d2);

    torch::Tensor d1 = up1->forward(d2);        // [B, c1, H, W]
    d1 = torch::cat({d1, e1}, 1);
    d1 = dec1->forward(d1);

    return head->forward(d1);                   // [B, numClasses, H, W]
}

// ---------------------------------------------------------------------------

CnnSegmenter::CnnSegmenter()
{
    // CPU only — LibTorch Metal/MPS support is unreliable; patch-based training
    // on a small net is feasible on CPU.
    device = torch::Device(torch::kCPU);
}

bool CnnSegmenter::train(const std::vector<CnnSlice> &slices, int numClasses,
                         int epochs, int patchSize, int batchSize, double &lastLoss)
{
    if (slices.empty() || numClasses < 2)
        return false;

    numClasses_ = numClasses;
    model = UNet(1, numClasses, 16);
    model->to(device);
    model->train();

    torch::optim::Adam optimizer(model->parameters(),
                                 torch::optim::AdamOptions(1e-3));

    // --- Class-balanced loss weights ---
    // Fossil pixels are far rarer than background, so plain cross-entropy
    // collapses toward the majority class (the model under-segments). Count
    // each class's labelled pixels across all training slices and weight the
    // loss inversely to frequency, so rare classes get proportionally more
    // influence. Weights are normalised to mean ~1 for a stable learning rate.
    std::vector<double> classCounts(static_cast<size_t>(numClasses), 0.0);
    for (const CnnSlice &cs : slices)
    {
        torch::Tensor lab = cs.labels;
        for (int c = 0; c < numClasses; c++)
            classCounts[static_cast<size_t>(c)] +=
                (lab == static_cast<int64_t>(c)).sum().item<double>();
    }
    double totalCounted = 0.0;
    for (double v : classCounts) totalCounted += v;

    torch::Tensor classWeights = torch::ones({numClasses}, torch::kFloat32);
    if (totalCounted > 0.0)
    {
        auto wAcc = classWeights.accessor<float, 1>();
        double sumW = 0.0;
        for (int c = 0; c < numClasses; c++)
        {
            // inverse frequency; guard against empty classes
            double freq = classCounts[static_cast<size_t>(c)] / totalCounted;
            double w = (freq > 0.0) ? (1.0 / freq) : 0.0;
            wAcc[c] = static_cast<float>(w);
            sumW += w;
        }
        // normalise so the mean weight is ~1 (keeps loss scale/LR sensible)
        if (sumW > 0.0)
        {
            float scale = static_cast<float>(numClasses) / static_cast<float>(sumW);
            classWeights = classWeights * scale;
        }
        qDebug() << "CNN class counts (per class):";
        for (int c = 0; c < numClasses; c++)
            qDebug() << "  class" << c << "count" << classCounts[static_cast<size_t>(c)]
                     << "weight" << classWeights[c].item<float>();
    }
    classWeights = classWeights.to(device);

    auto lossFn = torch::nn::CrossEntropyLoss(
        torch::nn::CrossEntropyLossOptions()
            .ignore_index(static_cast<int64_t>(CnnData::kIgnoreIndex))
            .weight(classWeights));

    lastLoss = 0.0;
    int realIters = 0;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        torch::Tensor images, labels;
        if (!CnnData::SamplePatchBatch(slices, patchSize, batchSize, images, labels))
            continue;

        images = images.to(device);
        labels = labels.to(device);

        optimizer.zero_grad();
        torch::Tensor logits = model->forward(images);   // [B, C, P, P]
        torch::Tensor loss = lossFn(logits, labels);      // ignores kIgnoreIndex

        if (std::isnan(loss.item<double>()))
            continue; // skip degenerate batch (e.g. all-ignore)

        loss.backward();
        optimizer.step();

        lastLoss = loss.item<double>();
        realIters++;

        if (epoch % 20 == 0)
            qDebug() << "CNN train iter" << epoch << "loss" << lastLoss;
    }

    trained = (realIters > 0);
    return trained;
}

torch::Tensor CnnSegmenter::predict(const torch::Tensor &image)
{
    // image: [1, H, W]. U-Net has 2 pool levels, so pad H,W up to a multiple
    // of 4, run, then crop back.
    torch::NoGradGuard noGrad;
    model->eval();

    int H = static_cast<int>(image.size(1));
    int W = static_cast<int>(image.size(2));
    int padH = (4 - (H % 4)) % 4;
    int padW = (4 - (W % 4)) % 4;

    torch::Tensor x = image.unsqueeze(0).to(device); // [1,1,H,W]
    if (padH || padW)
    {
        // pad = {left, right, top, bottom}
        x = torch::constant_pad_nd(x, {0, padW, 0, padH}, 0);
    }

    torch::Tensor logits = model->forward(x);        // [1, C, H', W']
    torch::Tensor probs = torch::softmax(logits, 1); // over classes
    probs = probs.squeeze(0);                        // [C, H', W']

    // Crop back to original size.
    probs = probs.index({torch::indexing::Slice(),
                         torch::indexing::Slice(0, H),
                         torch::indexing::Slice(0, W)});
    return probs.contiguous();
}
