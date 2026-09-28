/**
 * @file
 * Source: CNN Interface
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 */

// torch first (before any Qt header).
#include "cnninterface.h"
#include "cnndata.h"

#include "globals.h"
#include "display.h"
#include "fileio.h"
#include "mlupdateblockingdialog.h"

#include <QListWidget>
#include <QByteArray>
#include <QDebug>
#include <QApplication>

CnnInterface::CnnInterface()
    : segmenter(std::make_unique<CnnSegmenter>())
{
}

void CnnInterface::Generate(QListWidget *sliceSelector,
                            int samplePercent, int epochs,
                            int patchSize, int batchSize)
{
    if (FileCount == 0) return;
    if (SegmentCount < 2)
    {
        Message("CNN segmentation needs at least two segments.");
        return;
    }

    WriteAllData(CurrentFile);

    MLUpdateBlockingDialog::showDialog(mainwin, "", "", "CNN segmentation");

    // --- Gather training data from locked pixels across selected slices ---
    MLUpdateBlockingDialog::updateHighLevelText("Gathering training data");
    std::vector<CnnSlice> slices;
    long labelled = CnnData::GatherTrainingSlices(sliceSelector, samplePercent, slices);
    qDebug() << "CNN training: gathered" << labelled << "labelled pixels across"
             << static_cast<int>(slices.size()) << "slices";

    if (labelled == 0 || slices.empty())
    {
        MLUpdateBlockingDialog::hideDialog();
        Message("CNN training needs locked pixels on at least two segments. "
                "Paint a training sample with the segment brush (locks on).");
        return;
    }

    // --- Train the U-Net ---
    MLUpdateBlockingDialog::updateHighLevelText("Training CNN");
    double lastLoss = 0.0;
    bool ok = segmenter->train(slices, SegmentCount, epochs, patchSize, batchSize, lastLoss);
    qDebug() << "CNN training done, ok =" << ok << "final loss" << lastLoss;
    if (!ok)
    {
        MLUpdateBlockingDialog::hideDialog();
        Message("CNN training failed (not enough usable training patches).");
        return;
    }

    // --- Classify every selected slice and write into GA[] ---
    for (int i = 0; i < Files.count(); i++)
    {
        if (!(sliceSelector->item(i))->isSelected()) continue;
        if (MLUpdateBlockingDialog::isCancelled()) break;

        MLUpdateBlockingDialog::updateHighLevelText(QString("Segmenting slice %1").arg(i));

        LoadLocks(i);
        LoadMasks(i);
        for (int s = 0; s < SegmentCount; s++)
            LoadGreyData(i, s);

        // Build the input image tensor for this slice (grey/255).
        QImage src = ColArray.convertToFormat(QImage::Format_RGB32);
        const uchar *bits = src.bits();
        const int bpl = src.bytesPerLine();
        torch::Tensor image = torch::empty({1, fheight, fwidth}, torch::kFloat32);
        {
            auto acc = image.accessor<float, 3>();
            for (int y = 0; y < fheight; y++)
                for (int x = 0; x < fwidth; x++)
                    acc[0][y][x] = bits[y * bpl + x * 4 + 2] / 255.0f;
        }

        // Predict per-pixel class probabilities [C, H, W].
        torch::Tensor probs = segmenter->predict(image).cpu().contiguous();
        auto pAcc = probs.accessor<float, 3>();

        QByteArray locks = DoMaskLocking();

        // Write probs*255 into each segment's working image, with the ML
        // convention of a 128 floor for the winning segment (no black pixels).
        QVector<uchar *> outRows(SegmentCount);
        for (int y = 0; y < fheight; y++)
        {
            for (int s = 0; s < SegmentCount; s++)
                outRows[s] = GA[s]->scanLine(y);

            for (int x = 0; x < fwidth; x++)
            {
                if (locks[fwidth * y + x]) continue; // never overwrite locked/hidden

                int high = -1, highSeg = -1;
                for (int s = 0; s < SegmentCount; s++)
                {
                    int v = static_cast<int>(pAcc[s][y][x] * 255.0f + 0.5f);
                    if (v < 0) v = 0;
                    if (v > 255) v = 255;
                    if (v > high) { high = v; highSeg = s; }
                    outRows[s][x] = static_cast<uchar>(v);
                }
                if (high < 128 && highSeg >= 0)
                    outRows[highSeg][x] = static_cast<uchar>(128);
            }
        }

        for (int s = 0; s < SegmentCount; s++)
            SaveGreyData(i, s);
    }

    LoadAllData(CurrentFile);
    MLUpdateBlockingDialog::hideDialog();
}
