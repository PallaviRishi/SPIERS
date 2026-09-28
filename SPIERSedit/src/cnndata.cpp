/**
 * @file
 * Source: CNN Data Pipeline
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 */

// torch first (before any Qt header) — see cnndata.h note.
#include "cnndata.h"

#include "globals.h"
#include "display.h"
#include "fileio.h"
#include "mainwindow.h"
#include "labelledpoint.h"

#include <QListWidget>
#include <QImage>
#include <QList>
#include <QHash>

#include <random>

namespace CnnData
{

long GatherTrainingSlices(QListWidget *sliceSelector, int percentage,
                          std::vector<CnnSlice> &out)
{
    out.clear();

    // Reuse the existing locked-pixel labelling used by the ML system.
    // A locked pixel's class = segment with the highest GA value (>=128).
    QList<LabelledPoint> points = GenerateLabels(mainwin, percentage);
    if (points.isEmpty())
        return 0;

    // Group labelled points by originating slice.
    QHash<int, QList<const LabelledPoint *>> bySlice;
    for (const LabelledPoint &p : points)
        bySlice[p.z].append(&p);

    long totalLabelled = 0;

    const QList<int> sliceIds = bySlice.keys();
    for (int sliceId : sliceIds)
    {
        // Load the source (colour/greyscale) image for this slice.
        LoadColourData(sliceId);
        QImage src = ColArray.convertToFormat(QImage::Format_RGB32);
        const uchar *bits = src.bits();
        const int bpl = src.bytesPerLine();

        // Input image tensor [1, H, W], greyscale normalised 0..1
        // (grey = red channel; r==g==b on greyscale CT, and a reasonable
        //  luminance proxy for colour data at this stage).
        torch::Tensor image = torch::empty({1, fheight, fwidth}, torch::kFloat32);
        auto imgAcc = image.accessor<float, 3>();
        for (int y = 0; y < fheight; y++)
            for (int x = 0; x < fwidth; x++)
                imgAcc[0][y][x] = bits[y * bpl + x * 4 + 2] / 255.0f;

        // Label tensor [H, W], default ignore, filled at labelled pixels.
        // Use int64_t (not long): torch::kLong tensors are int64_t, and the
        // accessor is only instantiated for int64_t in libtorch.
        torch::Tensor labels = torch::full({fheight, fwidth},
                                           static_cast<int64_t>(kIgnoreIndex),
                                           torch::kLong);
        auto labAcc = labels.accessor<int64_t, 2>();

        long labelled = 0;
        for (const LabelledPoint *p : bySlice[sliceId])
        {
            if (p->x < 0 || p->x >= fwidth || p->y < 0 || p->y >= fheight)
                continue;
            labAcc[p->y][p->x] = static_cast<int64_t>(p->segment);
            labelled++;
        }

        if (labelled == 0)
            continue;

        CnnSlice cs;
        cs.image = image;
        cs.labels = labels;
        cs.sliceIndex = sliceId;
        cs.labelledPixels = labelled;
        out.push_back(std::move(cs));
        totalLabelled += labelled;
    }

    // Restore the current slice for normal operation.
    LoadAllData(CurrentFile);

    return totalLabelled;
}

bool SamplePatchBatch(const std::vector<CnnSlice> &slices, int patchSize,
                      int batchSize, torch::Tensor &images, torch::Tensor &labels)
{
    if (slices.empty() || patchSize <= 0 || batchSize <= 0)
        return false;

    images = torch::zeros({batchSize, 1, patchSize, patchSize}, torch::kFloat32);
    labels = torch::full({batchSize, patchSize, patchSize},
                         static_cast<int64_t>(kIgnoreIndex), torch::kLong);

    static std::mt19937 rng(1234);

    int produced = 0;
    int attempts = 0;
    const int maxAttempts = batchSize * 50;

    while (produced < batchSize && attempts < maxAttempts)
    {
        attempts++;

        // Pick a random slice, weighted by nothing fancy for now (uniform).
        const CnnSlice &cs = slices[rng() % slices.size()];
        int H = static_cast<int>(cs.image.size(1));
        int W = static_cast<int>(cs.image.size(2));
        if (H < patchSize || W < patchSize)
        {
            // Slice smaller than patch in some dim — skip (handled by caller
            // choosing an appropriate patch size; guard anyway).
            continue;
        }

        int oy = rng() % (H - patchSize + 1);
        int ox = rng() % (W - patchSize + 1);

        torch::Tensor imgPatch = cs.image.index(
            {0,
             torch::indexing::Slice(oy, oy + patchSize),
             torch::indexing::Slice(ox, ox + patchSize)});
        torch::Tensor labPatch = cs.labels.index(
            {torch::indexing::Slice(oy, oy + patchSize),
             torch::indexing::Slice(ox, ox + patchSize)});

        // Only keep patches that actually contain some supervision.
        torch::Tensor hasLabel = (labPatch != static_cast<int64_t>(kIgnoreIndex)).any();
        if (!hasLabel.item<bool>())
            continue;

        // Augmentation: random horizontal/vertical flip and 0/90/180/270 rot.
        int flipH = rng() % 2;
        int flipV = rng() % 2;
        int rot = rng() % 4;

        torch::Tensor ip = imgPatch.clone();
        torch::Tensor lp = labPatch.clone();
        if (flipH) { ip = ip.flip({1}); lp = lp.flip({1}); }
        if (flipV) { ip = ip.flip({0}); lp = lp.flip({0}); }
        if (rot)   { ip = ip.rot90(rot, {0, 1}); lp = lp.rot90(rot, {0, 1}); }

        images[produced][0] = ip;
        labels[produced] = lp;
        produced++;
    }

    if (produced == 0)
        return false;

    // If we produced fewer than requested (sparse labels), trim the batch.
    if (produced < batchSize)
    {
        images = images.narrow(0, 0, produced).clone();
        labels = labels.narrow(0, 0, produced).clone();
    }

    return true;
}

} // namespace CnnData
