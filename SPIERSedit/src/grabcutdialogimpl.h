/**
 * @file
 * Header: GrabCut Dialog Implementation
 *
 * Provides the interactive scribble painting dialog for GrabCut segmentation.
 *
 * Two classes are defined here:
 *
 *   GrabCutCanvas  — A QWidget subclass that handles mouse events for scribble
 *                    painting and renders the composite display (colour image +
 *                    scribble overlay + alpha mask overlay).
 *
 *   GrabCutDialogImpl — The QDialog that hosts GrabCutCanvas, all controls,
 *                       and the propagation UI. Owns the GrabCut and
 *                       SlicePropagation instances and runs them on a QThread.
 *
 * Integration with SPIERSedit:
 *   - Reads ColArray (the current colour source image) directly.
 *   - On Accept, writes alphaAsGAImage() into GA[CurrentSegment] and marks
 *     the slice dirty via FilesDirty[CurrentFile].
 *   - On propagation completion, updates all affected GA[] images and redraws.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2024 by the SPIERS contributors.
 */

#ifndef GRABCUTDIALOGIMPL_H
#define GRABCUTDIALOGIMPL_H

#include <QDialog>
#include <QWidget>
#include <QThread>
#include <QImage>
#include <QByteArray>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QResizeEvent>
#include <QPoint>
#include <QTimer>

#include "../ui/ui_grabcut.h"
#include "grabcut.h"
#include "slicepropagation.h"
#include "globals.h"

// ─── GrabCutCanvas ────────────────────────────────────────────────────────────

/**
 * @brief Interactive painting canvas for GrabCut scribbles.
 *
 * Displays the source colour image scaled to fit the widget, with two
 * optional overlays:
 *   1. Scribble overlay: green = foreground marks, red = background marks.
 *   2. Alpha overlay:    green tint on foreground pixels, red tint on background.
 *
 * Mouse interaction:
 *   - Left button drag   → paint foreground (TRIMAP_FOREGROUND)
 *   - Right button drag  → paint background (TRIMAP_BACKGROUND)
 *   - Middle button drag → erase to TRIMAP_UNKNOWN
 *   - Ctrl + left drag   → erase to TRIMAP_UNKNOWN
 *
 * All coordinates are automatically mapped between widget pixels and image pixels.
 */
class GrabCutCanvas : public QWidget
{
    Q_OBJECT

public:
    explicit GrabCutCanvas(QWidget *parent = nullptr);

    /** Set the source image to display (a copy is kept internally). */
    void setSourceImage(const QImage &img);

    /** Set the trimap byte array; must match image dimensions. */
    void setTrimap(const QByteArray &tm, int w, int h);

    /** Set the current alpha result for the overlay display. */
    void setAlpha(const std::vector<uchar> &alpha);

    /** Toggle scribble overlay visibility. */
    void setShowScribbles(bool show);

    /** Toggle alpha overlay visibility. */
    void setShowAlpha(bool show);

    /** Set the alpha overlay opacity (0–100). */
    void setAlphaOpacity(int opacity);

    /** Set the brush radius in image pixels. */
    void setBrushSize(int radius);

    /** Return the current trimap (flat byte array, image coords, row-major). */
    const QByteArray &trimap() const { return trimap_; }

    /** Clear all scribbles (reset entire trimap to TRIMAP_UNKNOWN). */
    void clearScribbles();

    /** Repaint the canvas from updated alpha or trimap data. */
    void refresh();

signals:
    /** Emitted after any scribble stroke completes (mouse release). */
    void scribbleCompleted();

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    QImage  sourceImage_;        ///< Original colour image (image coords)
    QImage  displayCache_;       ///< Composited image cached for fast repaint
    QByteArray trimap_;          ///< Current trimap (image coords, flat)
    std::vector<uchar> alpha_;   ///< Current alpha result (image coords)

    int imgW_ = 0, imgH_ = 0;
    int brushRadius_ = 8;
    bool showScribbles_ = true;
    bool showAlpha_     = true;
    int  alphaOpacity_  = 50;   ///< 0–100

    bool  painting_   = false;
    uchar paintValue_ = TRIMAP_FOREGROUND;

    bool   displayDirty_ = true;  ///< True when displayCache_ needs rebuild

    /// Map a widget-space point to image-space point (nearest pixel, clamped)
    QPoint widgetToImage(const QPoint &pt) const;

    /// Paint a circle of radius brushRadius_ at image coords (cx, cy)
    void paintCircle(int cx, int cy, uchar value);

    /// Rebuild the composited displayCache_ from sourceImage_ + overlays
    void rebuildDisplayCache();
};

// ─── GrabCutDialogImpl ────────────────────────────────────────────────────────

/**
 * @brief Full GrabCut dialog: scribble canvas + controls + propagation.
 *
 * Opened from the Segments menu → "Auto-segment (GrabCut)...".
 * The segment index to write into is passed at construction time.
 */
class GrabCutDialogImpl : public QDialog, public Ui::GrabCutDialog
{
    Q_OBJECT

public:
    /**
     * @param segmentIndex  Index into GA[] / Segments[] to write result into.
     * @param parent        Parent widget.
     */
    explicit GrabCutDialogImpl(int segmentIndex,
                                QWidget *parent = nullptr,
                                Qt::WindowFlags f = Qt::WindowFlags());
    ~GrabCutDialogImpl() override;

    /** True if the user accepted and a valid result was written to GA[]. */
    bool accepted() const { return accepted_; }

private slots:
    // ── Button handlers ──────────────────────────────────────────────────────
    void on_RunButton_clicked();
    void on_RefineButton_clicked();
    void on_ClearScribblesButton_clicked();
    void on_PropagateButton_clicked();
    void on_CancelPropButton_clicked();
    void on_buttonBox_accepted();
    void on_buttonBox_rejected();

    // ── Canvas events ─────────────────────────────────────────────────────────
    void onScribbleCompleted();

    // ── Control changes ───────────────────────────────────────────────────────
    void on_BrushSizeSpinBox_valueChanged(int v);
    void on_ShowScribblesCheckBox_toggled(bool checked);
    void on_ShowAlphaCheckBox_toggled(bool checked);
    void on_AlphaOpacitySlider_valueChanged(int v);

    // ── Propagation signals ───────────────────────────────────────────────────
    void onPropagationProgress(int percent, int sliceIdx);
    void onPropagationComplete(std::vector<PropagationResult> results, bool cancelled);

private:
    int  segmentIndex_;
    bool accepted_ = false;

    // ── Core algorithm objects ────────────────────────────────────────────────
    GrabCut       grabCut_;
    GrabCutState  currentState_;
    bool          stateValid_ = false;   ///< True once at least one run has completed

    // ── Propagation thread ────────────────────────────────────────────────────
    QThread            *propThread_  = nullptr;
    SlicePropagation   *propagation_ = nullptr;

    // ── Stored per-slice trimaps (to persist scribbles across slices) ─────────
    // Index matches Files[] / GA[] slice indices.
    std::vector<QByteArray> sliceTrimaps_;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /** Initialise everything from the current ColArray and CurrentFile. */
    void initialiseFromCurrentSlice();

    /** Run the GrabCut algorithm with current parameters and update the canvas. */
    void runGrabCut(int iterations);

    /** Write the current result into GA[segmentIndex_] and mark dirty. */
    void commitResultToGA();

    /** Set UI enabled state during/after propagation. */
    void setPropagationRunning(bool running);

    /** Build a summary string from propagation results and show in StatusLabel. */
    void showPropagationSummary(const std::vector<PropagationResult> &results);

    /** Save the current canvas trimap into sliceTrimaps_[CurrentFile]. */
    void saveCurrentTrimap();
};

#endif // GRABCUTDIALOGIMPL_H
