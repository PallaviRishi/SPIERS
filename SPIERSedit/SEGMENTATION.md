# GMM-based Automated Segmentation for SPIERSedit

## Overview

This branch adds a "GMM" generation tab to SPIERSedit that classifies fossil tomogram pixels as foreground (fossil) or background (matrix) using Gaussian Mixture Models. It sits alongside the existing Linear, Poly, Range, LCE, and Radial tabs in the generation toolbox and follows the same workflow: paint some locked training pixels, switch to the GMM tab, click Generate.

The method is implemented in C++ with no external dependencies beyond Qt (core + gui). It integrates directly into the existing generation pipeline — the output writes into GA[] like every other generation mode, so all manual editing tools continue to work for correcting remaining errors.

## Method

1. **User paints locked pixels** on at least two segments using the existing segment brush (with "segment brush applies locks" enabled). These become the GMM training data.
2. **GMM initialisation**: A 5-component Gaussian Mixture Model is fitted to the foreground segment's colour distribution and another to the background segment's, using k-means++ seeding followed by hard k-means clustering.
3. **Classification**: Every unlocked pixel is classified by comparing its RGB colour likelihood under the foreground GMM vs the background GMM. The class with higher probability wins.
4. **Output**: Foreground pixels are set to 255 in GA[], background to 0. The user can manually correct remaining errors with the brush tools.

## Results

### Hairy Ball dataset (vs manually-segmented binary ground truth)

324 slices, 624 x 710 pixels, 24-bit colour BMPs.

| Metric | Value |
|--------|-------|
| Accuracy | 96.1% |
| Precision | 87.3% |
| Recall | 95.6% |
| IoU | 83.9% |
| Avg runtime | 106 ms/slice |

### Kenostrychus dataset (vs SPIERSedit-generated segment files)

81 slices, 336 x 344 pixels (downsampled to 168 x 172), 24-bit colour BMPs.

| Metric | Value |
|--------|-------|
| Accuracy | 98.3% |
| Precision | 92.4% |
| Recall | 97.3% |
| IoU | 90.1% |
| Avg runtime | 6 ms/slice |

### Cross-specimen summary

| Dataset | Accuracy | IoU | Runtime |
|---------|----------|-----|---------|
| Hairy Ball (624x710) | 96.1% | 83.9% | 106 ms/slice |
| Kenostrychus (168x172) | 98.3% | 90.1% | 6 ms/slice |

### Comparison with existing methods

| Method | Accuracy | Notes |
|--------|----------|-------|
| SPIERSedit linear threshold (default, no tuning) | 59% | Fails when fossil is darker than background |
| SPIERSedit linear (manually inverted + optimal threshold) | 93-99% | Requires user to find inversion and correct threshold |
| **GMM tab (this implementation)** | **96-98%** | Automatic from locked pixels, no threshold tuning |

## Usage

1. Create at least two segments (e.g. "fossil" and "matrix")
2. Enable **"Segment brush applies locks"** (Brush menu)
3. Paint a few strokes on known fossil pixels with the fossil segment selected
4. Paint a few strokes on known background pixels with the matrix segment selected
5. Switch to the **GMM** tab in the Generation toolbox
6. Select target slices in the Slice Selector
7. Click **Generate**

The result appears immediately. Use the brightness brush or segment brush to correct any remaining errors, then re-generate if needed.

## Files

| File | Purpose |
|------|---------|
| `src/gmm.h`, `src/gmm.cpp` | 5-component GMM engine (k-means++ init, EM) |
| `src/grabcut.h`, `src/grabcut.cpp` | GMM classification wrapper |
| `src/gmmautosample.h`, `src/gmmautosample.cpp` | Auto-sample generator for RF integration |

## Integration with the ML (Random Forest) system

The `gmmautosample.h/cpp` module provides automatic training sample generation for the Random Forest pipeline on the `Qt6-and-AI` branch.

### The problem it solves

The RF system requires the user to manually paint thousands of labelled training pixels across multiple slices. The GMM can automate this: a few scribbles on one slice produces 20,000+ high-confidence labelled points in 100ms.

### How to integrate

Copy `gmm.h/cpp`, `grabcut.h/cpp`, and `gmmautosample.h/cpp` into the Qt6-and-AI branch's `src/` folder and add them to the `.pro`. Then call `gmmAutoSample()` from `MLInterface`:

```cpp
#include "gmmautosample.h"

void MLInterface::AutoSampleFromGMM()
{
    QByteArray trimap = buildTrimapFromLocks(CurrentFile, 0, 1);

    QVector<GmmLabelledPoint> autoLabels = gmmAutoSample(
        CurrentFile, ColArray, trimap,
        0,       // foreground segment
        1,       // background segment
        3.0,     // confidence threshold
        20000    // max points per class
    );

    for (const auto &pt : autoLabels)
        labels.append(LabelledPoint(pt.x, pt.y, pt.z, pt.segment));

    Train(false);
}
```

### Why both approaches complement each other

| | GMM | Random Forest |
|---|---|---|
| Features | RGB colour (3 dimensions) | 18+ feature types (texture, gradient, tensor) |
| Speed | 100ms per slice | Seconds to hours (feature precalculation) |
| Best for | Clear colour separation | Textural ambiguity |
| Weakness | No spatial context | Needs large labelled sample |

The GMM handles the 96% of pixels where colour is decisive. The RF handles the remaining 4% where texture and context matter. Together they reduce manual effort to near-zero.

## Building

Requires Qt 5.x or 6.x (core + gui modules only). No OpenCV or other external dependencies.

```bash
cd SPIERSedit
qmake SPIERSedit.pro
make -j$(nproc)
```
