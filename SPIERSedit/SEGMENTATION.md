# GMM-based Automated Segmentation for SPIERSedit

## Overview

This branch adds an automated segmentation method to SPIERSedit that classifies fossil tomogram pixels as foreground (fossil) or background (matrix) using Gaussian Mixture Models (GMMs). It replaces the need for manual brightness threshold adjustment with a scribble-based interactive approach that learns the colour distribution of each class directly from user annotations.

The method is implemented entirely in C++ with no external dependencies beyond Qt 5.x, and integrates directly into the existing SPIERSedit workflow — the output writes into the same GA[] greyscale image system, so all existing manual editing tools (brighten brush, segment brush, masks, locks) continue to work for correcting the remaining errors.

## Method

1. **User paints scribbles** on the source image: green for definite foreground, red for definite background.
2. **GMM initialisation**: A 5-component Gaussian Mixture Model is fitted to the foreground scribble pixels and another to the background scribble pixels, using k-means++ seeding followed by hard k-means clustering to establish distinct components with proper covariance estimates.
3. **Classification**: Every unscribbled pixel is classified by comparing its RGB colour likelihood under the foreground GMM vs the background GMM. The class with higher probability wins.
4. **Output**: Foreground pixels are set to 255 in GA[], background to 0. The user can then manually correct any remaining errors with the existing brush tools.

### Inter-slice propagation

Once a single slice is segmented, the fitted GMMs can be propagated to adjacent slices as a warm start — the algorithm re-runs on each neighbouring slice using the previous slice's colour model. This means users only need to draw scribbles on a small number of slices rather than annotating every one individually.

## Results

Tested on the "Hairy Ball" fossil dataset (324 slices, 624 x 710 pixels, 24-bit colour BMPs). Ground truth is a fully manually-segmented binary mask set (0 = background, 255 = fossil) produced by hand-editing every pixel.

### Single slice (slice 150, representative mid-stack)

| Metric | Value |
|--------|-------|
| Accuracy | 96.7% |
| Precision | 87.3% |
| Recall | 95.6% |
| Runtime | 102 ms |

### Full stack (33 slices sampled, every 10th)

| Metric | Value |
|--------|-------|
| Accuracy | 96.1% |
| Precision | 87.3% |
| Recall | 95.6% |
| IoU | 83.9% |
| Avg runtime | 102 ms/slice |

Per-slice accuracy is consistently 91–99% across the full stack with no collapse on edge slices.

The 87% precision indicates slight over-segmentation at boundaries — some edge pixels are classified as fossil that the manual segmenter excluded. This is correctable with a few seconds of brush editing.

### Comparison with existing SPIERSedit linear threshold

| Method | Accuracy | Notes |
|--------|----------|-------|
| SPIERSedit default (linear brightness threshold, no manual tuning) | 59% | Default expects brighter = fossil; this specimen is darker than background |
| SPIERSedit with manual inversion + optimal threshold | 93–99% | Requires user to discover inversion is needed and find the right threshold |
| **This GMM method (from scribbles)** | **96.1%** | Automatic — no threshold tuning, no inversion knowledge required |

The key advantage is not marginal accuracy improvement but workflow improvement: the user paints a few scribbles and gets a good result in 100ms, rather than needing to understand and manually adjust RGB weights, inversion, and global slider position per-slice.

## Files

| File | Purpose |
|------|---------|
| `src/gmm.h`, `src/gmm.cpp` | 5-component GMM with k-means++ init and EM |
| `src/grabcut.h`, `src/grabcut.cpp` | Segmentation engine (GMM classification) |
| `src/grabcutdialogimpl.h`, `src/grabcutdialogimpl.cpp` | Interactive scribble painting dialog |
| `src/grabcutglobals.h`, `src/grabcutglobals.cpp` | Alpha cache for the generation pipeline |
| `src/grabcutpersistence.h`, `src/grabcutpersistence.cpp` | Save/load trimap and GMM state to disk |
| `src/slicepropagation.h`, `src/slicepropagation.cpp` | Inter-slice propagation engine |
| `ui/grabcut.ui` | Qt Designer dialog layout |

## Usage

1. Open a dataset in SPIERSedit
2. Select a segment in the Segments panel
3. Go to **Segments > Auto-segment (GrabCut)...**
4. Paint green scribbles on fossil regions, red on background
5. Click **Run GrabCut**
6. Inspect the result overlay; add more scribbles and click **Refine** if needed
7. Click **OK** to write the result into the segment
8. Optionally use the propagation panel to extend to adjacent slices

## Building

Requires Qt 5.x (tested with 5.15.2 via Homebrew on macOS). No VTK or other dependencies needed for SPIERSedit.

```bash
cd SPIERSedit
/path/to/qt5/bin/qmake SPIERSedit.pro
make -j$(nproc)
```

The binary is produced at `SPIERSedit/bin/SPIERSedit64`.


## Integration with the ML (Random Forest) system

The GMM module includes `gmmautosample.h/cpp` which provides automatic training sample generation for the Random Forest pipeline on the `Qt6-and-AI` branch.

### The problem it solves

The RF system requires the user to manually paint thousands of labelled training pixels across multiple slices (step 4 of the ML workflow). This is the most tedious and time-consuming part of the ML pipeline.

### How it works

`gmmAutoSample()` takes a few user scribbles, runs GMM classification (100ms), and exports high-confidence pixels as labelled training points in the format expected by `MLInterface::Sample()`. Only pixels where the GMM is confident (likelihood ratio > threshold) are included, ensuring the RF gets clean training data.

### Integration into MLInterface

To add a "Auto-sample from GMM" button to the ML tab:

```cpp
#include "gmmautosample.h"

// In MLInterface — called when user clicks "Auto-sample from GMM":
void MLInterface::AutoSampleFromGMM()
{
    // Build trimap from current locks (user's initial scribbles)
    QByteArray trimap = buildTrimapFromLocks(CurrentFile, 0, 1);

    // Run GMM and get high-confidence labelled points
    QVector<GmmLabelledPoint> autoLabels = gmmAutoSample(
        CurrentFile,
        ColArray,
        trimap,
        0,              // foreground segment index
        1,              // background segment index
        3.0,            // confidence threshold
        20000           // max points per class
    );

    // Convert to LabelledPoint and append to training set
    for (const auto &pt : autoLabels)
        labels.append(LabelledPoint(pt.x, pt.y, pt.z, pt.segment));

    // Proceed with normal Train() and Generate()
    Train(false);
}
```

### Workflow with integration

1. User paints a few quick scribbles (5 seconds)
2. Click "Auto-sample from GMM" → generates 20,000+ clean training points instantly
3. RF trains on these points (seconds)
4. RF generates with spatial/textural features → handles edge cases the GMM missed
5. User inspects, adds corrections, retrains if needed

This eliminates the most tedious step (manual sample painting) while preserving the full power of the Random Forest for difficult cases.
