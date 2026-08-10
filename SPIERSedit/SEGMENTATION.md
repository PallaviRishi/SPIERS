# GMM-based Automated Segmentation for SPIERSedit

## Overview

This branch adds an automated segmentation method to SPIERSedit that classifies fossil tomogram pixels as foreground (fossil) or background (matrix) using Gaussian Mixture Models (GMMs). It replaces the need for manual brightness threshold adjustment with a scribble-based interactive approach that learns the colour distribution of each class directly from user annotations.

The method is implemented entirely in C++ with no external dependencies beyond Qt (core + gui), and integrates directly into the existing SPIERSedit workflow — the output writes into the same GA[] greyscale image system, so all existing manual editing tools (brighten brush, segment brush, masks, locks) continue to work for correcting the remaining errors.

## Method

1. **User paints scribbles** on the source image: green for definite foreground, red for definite background.
2. **GMM initialisation**: A 5-component Gaussian Mixture Model is fitted to the foreground scribble pixels and another to the background scribble pixels, using k-means++ seeding followed by hard k-means clustering to establish distinct components with proper covariance estimates.
3. **Classification**: Every unscribbled pixel is classified by comparing its RGB colour likelihood under the foreground GMM vs the background GMM. The class with higher probability wins.
4. **Output**: Foreground pixels are set to 255 in GA[], background to 0. The user can then manually correct any remaining errors with the existing brush tools.

### Inter-slice propagation

Once a single slice is segmented, the fitted GMMs can be propagated to adjacent slices as a warm start — the algorithm re-runs on each neighbouring slice using the previous slice's colour model. This means users only need to draw scribbles on a small number of slices rather than annotating every one individually.

## Results

### Hairy Ball dataset

324 slices, 624 x 710 pixels, 24-bit colour BMPs. Ground truth is a fully manually-segmented binary mask set (0 = background, 255 = fossil) produced by hand-editing every pixel.

| Metric | Single slice (150) | Full stack (33 sampled) |
|--------|-------------------|------------------------|
| Accuracy | 96.7% | 96.1% |
| Precision | 87.3% | 87.3% |
| Recall | 95.6% | 95.6% |
| IoU | — | 83.9% |
| Runtime | 102 ms | 102 ms/slice |

Per-slice accuracy is consistently 91–99% across the full stack. The 87% precision indicates slight over-segmentation at boundaries, correctable with a few seconds of brush editing.

### Kenostrychus dataset

81 slices, 336 x 344 pixels (downsampled to 168 x 172 for segmentation), 24-bit colour BMPs. Ground truth is the SPIERSedit-generated segment file with manual edits.

| Metric | Full stack (17 sampled) |
|--------|------------------------|
| Accuracy | 98.3% |
| Precision | 92.4% |
| Recall | 97.3% |
| IoU | 90.1% |
| Runtime | 5 ms/slice |

Per-slice accuracy ranges from 95–100%, with several slices achieving perfect classification. The smaller image size gives near-instant runtime.

### Cross-specimen summary

| Dataset | Accuracy | IoU | Runtime |
|---------|----------|-----|---------|
| Hairy Ball (624×710) | 96.1% | 83.9% | 102 ms/slice |
| Kenostrychus (168×172) | 98.3% | 90.1% | 5 ms/slice |

The method generalises well across different fossil specimens with different colour profiles and morphologies.

### Comparison with existing SPIERSedit linear threshold

| Method | Accuracy | Notes |
|--------|----------|-------|
| SPIERSedit default (linear brightness threshold, no tuning) | 59% | Fails when fossil is darker than background |
| SPIERSedit with manual inversion + optimal threshold | 93–99% | Requires user to find inversion and correct threshold |
| **This GMM method (from scribbles)** | **96–98%** | Automatic — no threshold tuning required |

## Files

| File | Purpose |
|------|---------|
| `src/gmm.h`, `src/gmm.cpp` | 5-component GMM with k-means++ init |
| `src/grabcut.h`, `src/grabcut.cpp` | Segmentation engine (GMM classification) |
| `src/gmmautosample.h`, `src/gmmautosample.cpp` | Auto-sample generator for RF integration |
| `src/grabcutdialogimpl.h`, `src/grabcutdialogimpl.cpp` | Interactive scribble painting dialog |
| `src/grabcutglobals.h`, `src/grabcutglobals.cpp` | Alpha cache for the generation pipeline |
| `src/grabcutpersistence.h`, `src/grabcutpersistence.cpp` | Save/load trimap and GMM state to disk |
| `src/slicepropagation.h`, `src/slicepropagation.cpp` | Inter-slice propagation engine |
| `ui/grabcut.ui` | Qt Designer dialog layout |

## Usage (standalone)

1. Open a dataset in SPIERSedit
2. Select a segment in the Segments panel
3. Go to **Segments > Auto-segment (GrabCut)...**
4. Paint green scribbles on fossil regions, red on background
5. Click **Run GrabCut**
6. Inspect the result overlay; add more scribbles and click **Refine** if needed
7. Click **OK** to write the result into the segment
8. Optionally use the propagation panel to extend to adjacent slices

## Integration with the ML (Random Forest) system

This module is designed to integrate with the Random Forest ML pipeline on the `Qt6-and-AI` branch as an automatic training sample generator.

### The problem it solves

The RF system (SPIERSedit v4, `Qt6-and-AI` branch) requires the user to manually paint thousands of labelled training pixels across multiple slices using locked segment brushes. This is step 4 of the ML workflow and the most tedious part of the pipeline.

The GMM module eliminates this step: instead of painting thousands of pixels, the user paints a few quick scribbles, the GMM classifies the entire slice in 100ms, and high-confidence pixels are exported as labelled training points for the Random Forest.

### How to integrate

The GMM segmentation is self-contained in 6 files with no dependencies beyond Qt core/gui:

```
src/gmm.h          src/gmm.cpp
src/grabcut.h       src/grabcut.cpp
src/gmmautosample.h src/gmmautosample.cpp
```

To add to the `Qt6-and-AI` branch:

1. Copy the 6 files into `SPIERSedit/src/`
2. Add them to `SPIERSedit.pro` HEADERS and SOURCES
3. Call `gmmAutoSample()` from `MLInterface` where the user would normally provide a manual sample

### Integration code

Add an "Auto-sample from GMM" button to the ML tab. When clicked:

```cpp
#include "gmmautosample.h"

void MLInterface::AutoSampleFromGMM()
{
    // Build trimap from current locks (user's few initial scribbles)
    QByteArray trimap = buildTrimapFromLocks(CurrentFile, 0, 1);

    // Run GMM and export high-confidence pixels as training points
    QVector<GmmLabelledPoint> autoLabels = gmmAutoSample(
        CurrentFile,
        ColArray,
        trimap,
        0,              // foreground segment index
        1,              // background segment index
        3.0,            // confidence threshold (only very confident pixels)
        20000           // max points per class
    );

    // Convert to LabelledPoint and append to ML training set
    for (const auto &pt : autoLabels)
        labels.append(LabelledPoint(pt.x, pt.y, pt.z, pt.segment));

    // Train the Random Forest on the auto-generated sample
    Train(false);
}
```

### Combined workflow

| Step | Without GMM | With GMM |
|------|-------------|----------|
| 1. Create segments | Same | Same |
| 2. Define features | Same | Same |
| 3. Precalculate features | Same | Same |
| 4. **Define training sample** | **Manually paint thousands of pixels across multiple slices (minutes)** | **Paint a few scribbles, click "Auto-sample from GMM" (seconds)** |
| 5. Train RF | Same | Same |
| 6. Generate | Same | Same |

The GMM replaces step 4 only. The Random Forest still provides the final classification using its spatial/textural features — the GMM just gives it a better starting sample, faster.

### Why both approaches complement each other

| | GMM | Random Forest |
|---|---|---|
| Features | RGB colour only (3 dimensions) | 18+ feature types (texture, gradient, tensor, multi-scale) |
| Training | Learns at runtime from scribbles | Learns from labelled sample |
| Speed | 100ms per slice | Seconds to hours (feature precalculation) |
| Best for | Clear colour separation (most fossil datasets) | Difficult cases with textural ambiguity |
| Weakness | No spatial context | Needs large labelled sample |

The GMM handles the 80% of pixels where colour alone is decisive. The RF handles the remaining 20% where local texture and context matter. Together they reduce manual effort to near-zero for most datasets.

## Building

Requires Qt 5.x or 6.x (core + gui modules only). No OpenCV or other external dependencies needed for the GMM module.

```bash
cd SPIERSedit
qmake SPIERSedit.pro
make -j$(nproc)
```

The binary is produced at `SPIERSedit/bin/SPIERSedit64`.
