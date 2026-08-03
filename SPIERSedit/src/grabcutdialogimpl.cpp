/**
 * @file
 * Source: GrabCut Dialog Implementation
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2024 by the SPIERS contributors.
 */

#include "grabcutdialogimpl.h"
#include "fileio.h"
#include "display.h"

#include <QPainter>
#include <QMessageBox>
#include <QApplication>
#include <QCursor>
#include <algorithm>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════════
// GrabCutCanvas
// ═══════════════════════════════════════════════════════════════════════════════

GrabCutCanvas::GrabCutCanvas(QWidget *parent)
    : QWidget(parent)
{
    setMouseTracking(true);
    setCursor(Qt::CrossCursor);
}

// ─── Data setters ─────────────────────────────────────────────────────────────

void GrabCutCanvas::setSourceImage(const QImage &img)
{
    sourceImage_ = img.convertToFormat(QImage::Format_RGB32);
    imgW_ = sourceImage_.width();
    imgH_ = sourceImage_.height();
    displayDirty_ = true;
}

void GrabCutCanvas::setTrimap(const QByteArray &tm, int w, int h)
{
    trimap_ = tm;
    imgW_   = w;
    imgH_   = h;
    displayDirty_ = true;
}

void GrabCutCanvas::setAlpha(const std::vector<uchar> &alpha)
{
    alpha_ = alpha;
    displayDirty_ = true;
}

void GrabCutCanvas::setShowScribbles(bool show)
{
    showScribbles_ = show;
    displayDirty_  = true;
    update();
}

void GrabCutCanvas::setShowAlpha(bool show)
{
    showAlpha_    = show;
    displayDirty_ = true;
    update();
}

void GrabCutCanvas::setAlphaOpacity(int opacity)
{
    alphaOpacity_ = opacity;
    displayDirty_ = true;
    update();
}

void GrabCutCanvas::setBrushSize(int radius)
{
    brushRadius_ = radius;
}

void GrabCutCanvas::clearScribbles()
{
    trimap_.fill(static_cast<char>(TRIMAP_UNKNOWN));
    displayDirty_ = true;
    update();
}

void GrabCutCanvas::refresh()
{
    displayDirty_ = true;
    update();
}

// ─── Coordinate mapping ───────────────────────────────────────────────────────

QPoint GrabCutCanvas::widgetToImage(const QPoint &pt) const
{
    if (imgW_ == 0 || imgH_ == 0) return QPoint(0, 0);

    int ww = width();
    int wh = height();

    // We scale the image to fit within the widget maintaining aspect ratio
    double scaleX = static_cast<double>(ww) / imgW_;
    double scaleY = static_cast<double>(wh) / imgH_;
    double scale  = std::min(scaleX, scaleY);

    int dispW = static_cast<int>(imgW_ * scale);
    int dispH = static_cast<int>(imgH_ * scale);
    int offX  = (ww - dispW) / 2;
    int offY  = (wh - dispH) / 2;

    int ix = static_cast<int>((pt.x() - offX) / scale);
    int iy = static_cast<int>((pt.y() - offY) / scale);

    ix = std::max(0, std::min(imgW_ - 1, ix));
    iy = std::max(0, std::min(imgH_ - 1, iy));
    return QPoint(ix, iy);
}

// ─── Scribble painting ────────────────────────────────────────────────────────

void GrabCutCanvas::paintCircle(int cx, int cy, uchar value)
{
    if (trimap_.isEmpty() || imgW_ == 0) return;

    int r = brushRadius_;
    for (int dy = -r; dy <= r; dy++)
    {
        for (int dx = -r; dx <= r; dx++)
        {
            if (dx * dx + dy * dy <= r * r)
            {
                int px = cx + dx;
                int py = cy + dy;
                if (px >= 0 && px < imgW_ && py >= 0 && py < imgH_)
                    trimap_[py * imgW_ + px] = static_cast<char>(value);
            }
        }
    }
    displayDirty_ = true;
}

// ─── Mouse events ─────────────────────────────────────────────────────────────

void GrabCutCanvas::mousePressEvent(QMouseEvent *event)
{
    painting_ = true;

    bool ctrl = event->modifiers() & Qt::ControlModifier;
    if (event->button() == Qt::LeftButton && !ctrl)
        paintValue_ = TRIMAP_FOREGROUND;
    else if (event->button() == Qt::RightButton)
        paintValue_ = TRIMAP_BACKGROUND;
    else // middle button or ctrl+left = erase
        paintValue_ = TRIMAP_UNKNOWN;

    QPoint ip = widgetToImage(event->pos());
    paintCircle(ip.x(), ip.y(), paintValue_);
    update();
}

void GrabCutCanvas::mouseMoveEvent(QMouseEvent *event)
{
    if (!painting_) return;

    QPoint ip = widgetToImage(event->pos());
    paintCircle(ip.x(), ip.y(), paintValue_);
    update();
}

void GrabCutCanvas::mouseReleaseEvent(QMouseEvent *event)
{
    Q_UNUSED(event);
    if (painting_)
    {
        painting_ = false;
        emit scribbleCompleted();
    }
}

void GrabCutCanvas::resizeEvent(QResizeEvent *event)
{
    Q_UNUSED(event);
    displayDirty_ = true;
}

// ─── Rendering ────────────────────────────────────────────────────────────────

void GrabCutCanvas::rebuildDisplayCache()
{
    if (sourceImage_.isNull()) return;

    int ww = width();
    int wh = height();
    if (ww == 0 || wh == 0) return;

    // Compute scale and offset for aspect-ratio-preserving fit
    double scaleX = static_cast<double>(ww) / imgW_;
    double scaleY = static_cast<double>(wh) / imgH_;
    double scale  = std::min(scaleX, scaleY);
    int dispW = static_cast<int>(imgW_ * scale);
    int dispH = static_cast<int>(imgH_ * scale);
    int offX  = (ww - dispW) / 2;
    int offY  = (wh - dispH) / 2;

    // Start with the scaled source image
    displayCache_ = QImage(ww, wh, QImage::Format_RGB32);
    displayCache_.fill(Qt::black);

    QImage scaled = sourceImage_.scaled(dispW, dispH,
                                        Qt::IgnoreAspectRatio,
                                        Qt::SmoothTransformation);
    QPainter p(&displayCache_);
    p.drawImage(offX, offY, scaled);

    // ── Alpha overlay ─────────────────────────────────────────────────────────
    if (showAlpha_ && !alpha_.empty() && alphaOpacity_ > 0)
    {
        // Build a small RGBA overlay image in image coords, then scale
        QImage alphaOverlay(imgW_, imgH_, QImage::Format_ARGB32);
        int    alpha8 = static_cast<int>(255.0 * alphaOpacity_ / 100.0);

        for (int y = 0; y < imgH_; y++)
        {
            for (int x = 0; x < imgW_; x++)
            {
                uchar a = alpha_[static_cast<size_t>(y * imgW_ + x)];
                if (a == ALPHA_FG)
                    alphaOverlay.setPixel(x, y, qRgba(0, 200, 0, alpha8));
                else
                    alphaOverlay.setPixel(x, y, qRgba(0, 0, 0, 0));
            }
        }
        QImage scaledAlpha = alphaOverlay.scaled(dispW, dispH,
                                                 Qt::IgnoreAspectRatio,
                                                 Qt::FastTransformation);
        p.drawImage(offX, offY, scaledAlpha);
    }

    // ── Scribble overlay ──────────────────────────────────────────────────────
    if (showScribbles_ && !trimap_.isEmpty())
    {
        QImage scribbleOverlay(imgW_, imgH_, QImage::Format_ARGB32);
        scribbleOverlay.fill(Qt::transparent);

        for (int y = 0; y < imgH_; y++)
        {
            for (int x = 0; x < imgW_; x++)
            {
                uchar t = static_cast<uchar>(trimap_.at(y * imgW_ + x));
                if (t == TRIMAP_FOREGROUND)
                    scribbleOverlay.setPixel(x, y, qRgba(0, 255, 0, 200));
                else if (t == TRIMAP_BACKGROUND)
                    scribbleOverlay.setPixel(x, y, qRgba(255, 0, 0, 200));
            }
        }
        QImage scaledScribble = scribbleOverlay.scaled(dispW, dispH,
                                                       Qt::IgnoreAspectRatio,
                                                       Qt::FastTransformation);
        p.drawImage(offX, offY, scaledScribble);
    }

    p.end();
    displayDirty_ = false;
}

void GrabCutCanvas::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);
    if (displayDirty_) rebuildDisplayCache();
    if (displayCache_.isNull()) return;

    QPainter p(this);
    p.drawImage(0, 0, displayCache_);
}

// ═══════════════════════════════════════════════════════════════════════════════
// GrabCutDialogImpl
// ═══════════════════════════════════════════════════════════════════════════════

GrabCutDialogImpl::GrabCutDialogImpl(int segmentIndex,
                                     QWidget *parent,
                                     Qt::WindowFlags f)
    : QDialog(parent, f)
    , segmentIndex_(segmentIndex)
{
    setupUi(this);
    setWindowIcon(QIcon(":/icons/ProgramIcon.bmp"));

    // Connect canvas signal
    connect(Canvas, &GrabCutCanvas::scribbleCompleted,
            this,   &GrabCutDialogImpl::onScribbleCompleted);

    // Initialise trimaps_ to hold one entry per slice
    sliceTrimaps_.resize(static_cast<size_t>(FileCount));

    // Set propagation range defaults to current slice
    PropFromSpinBox->setMaximum(FileCount - 1);
    PropToSpinBox->setMaximum(FileCount - 1);
    PropFromSpinBox->setValue(CurrentFile);
    PropToSpinBox->setValue(CurrentFile);

    // Update labels
    SliceLabel->setText(QString("Slice: %1").arg(CurrentFile + 1));
    if (segmentIndex_ >= 0 && segmentIndex_ < SegmentCount)
        SegmentLabel->setText(QString("Segment: %1")
                              .arg(Segments[segmentIndex_]->Name));

    initialiseFromCurrentSlice();
}

GrabCutDialogImpl::~GrabCutDialogImpl()
{
    // Stop propagation thread if running
    if (propThread_ && propThread_->isRunning())
    {
        if (propagation_) propagation_->cancel();
        propThread_->quit();
        propThread_->wait(3000);
    }
    delete propThread_;
    // propagation_ is owned by propThread_ (moveToThread), deleted with it
}

// ─── Initialisation ───────────────────────────────────────────────────────────

void GrabCutDialogImpl::initialiseFromCurrentSlice()
{
    // Get the source colour image — use ColArray (already loaded in globals)
    QImage colImg = ColArray.convertToFormat(QImage::Format_RGB32);
    if (colImg.isNull())
    {
        StatusLabel->setText("No colour image loaded.");
        return;
    }

    int W = colImg.width();
    int H = colImg.height();

    // Restore or create trimap for this slice
    QByteArray &tm = sliceTrimaps_[static_cast<size_t>(CurrentFile)];
    if (tm.isEmpty())
        tm = QByteArray(W * H, static_cast<char>(TRIMAP_UNKNOWN));

    // Set up canvas
    Canvas->setSourceImage(colImg);
    Canvas->setTrimap(tm, W, H);
    Canvas->setBrushSize(BrushSizeSpinBox->value());
    Canvas->setShowScribbles(ShowScribblesCheckBox->isChecked());
    Canvas->setShowAlpha(ShowAlphaCheckBox->isChecked());
    Canvas->setAlphaOpacity(AlphaOpacitySlider->value());

    // Set up GrabCut engine
    grabCut_.setImage(colImg);
    grabCut_.setTrimap(tm);
    grabCut_.gamma = GammaSpinBox->value();

    // If we have a previously saved state for this slice, restore the alpha
    if (stateValid_)
    {
        Canvas->setAlpha(currentState_.alpha);
    }

    Canvas->refresh();
    StatusLabel->setText("Draw foreground (green) and background (red) scribbles, then click Run GrabCut.");
}

// ─── Run GrabCut ─────────────────────────────────────────────────────────────

void GrabCutDialogImpl::runGrabCut(int iterations)
{
    saveCurrentTrimap();

    QByteArray &tm = sliceTrimaps_[static_cast<size_t>(CurrentFile)];

    // Check there's at least one fg and one bg scribble
    bool hasFG = false, hasBG = false;
    for (int i = 0; i < tm.size(); i++)
    {
        uchar t = static_cast<uchar>(tm.at(i));
        if (t == TRIMAP_FOREGROUND) hasFG = true;
        if (t == TRIMAP_BACKGROUND) hasBG = true;
        if (hasFG && hasBG) break;
    }

    if (!hasFG || !hasBG)
    {
        StatusLabel->setText("Please paint at least some foreground (green) AND "
                             "background (red) scribbles before running.");
        return;
    }

    QApplication::setOverrideCursor(Qt::WaitCursor);
    StatusLabel->setText("Running GrabCut...");
    ProgressBar->setValue(0);
    QApplication::processEvents();

    // Set up fresh engine state
    grabCut_.setTrimap(tm);
    grabCut_.gamma = GammaSpinBox->value();

    grabCut_.setProgressCallback([this](int pct)
    {
        ProgressBar->setValue(pct);
        QApplication::processEvents();
    });

    if (stateValid_)
        grabCut_.setState(currentState_);
    else
        grabCut_.initialise();  // seed from trimap scribbles

    grabCut_.run(iterations);

    currentState_ = grabCut_.getState();
    stateValid_   = true;

    // Update canvas overlay
    Canvas->setAlpha(grabCut_.alpha());
    Canvas->refresh();

    ProgressBar->setValue(100);
    StatusLabel->setText(QString("Done. FG pixels: %1")
                         .arg(std::count(grabCut_.alpha().begin(),
                                         grabCut_.alpha().end(), ALPHA_FG)));

    QApplication::restoreOverrideCursor();
}

// ─── Button handlers ──────────────────────────────────────────────────────────

void GrabCutDialogImpl::on_RunButton_clicked()
{
    stateValid_ = false;  // Force re-initialisation from current scribbles
    runGrabCut(IterationsSpinBox->value());
}

void GrabCutDialogImpl::on_RefineButton_clicked()
{
    runGrabCut(1);
}

void GrabCutDialogImpl::on_ClearScribblesButton_clicked()
{
    Canvas->clearScribbles();
    sliceTrimaps_[static_cast<size_t>(CurrentFile)] =
        QByteArray(grabCut_.width() * grabCut_.height(),
                   static_cast<char>(TRIMAP_UNKNOWN));
    stateValid_ = false;
    Canvas->setAlpha({});
    Canvas->refresh();
    StatusLabel->setText("Scribbles cleared.");
}

void GrabCutDialogImpl::onScribbleCompleted()
{
    // Sync canvas trimap back so we have the latest for the next run
    saveCurrentTrimap();
}

// ─── Control value changes ────────────────────────────────────────────────────

void GrabCutDialogImpl::on_BrushSizeSpinBox_valueChanged(int v)
{
    Canvas->setBrushSize(v);
}

void GrabCutDialogImpl::on_ShowScribblesCheckBox_toggled(bool checked)
{
    Canvas->setShowScribbles(checked);
}

void GrabCutDialogImpl::on_ShowAlphaCheckBox_toggled(bool checked)
{
    Canvas->setShowAlpha(checked);
}

void GrabCutDialogImpl::on_AlphaOpacitySlider_valueChanged(int v)
{
    Canvas->setAlphaOpacity(v);
}

// ─── Accept / Reject ─────────────────────────────────────────────────────────

void GrabCutDialogImpl::on_buttonBox_accepted()
{
    if (!stateValid_)
    {
        QMessageBox::warning(this, "No result",
                             "Run GrabCut at least once before accepting.");
        return;
    }
    commitResultToGA();
    accepted_ = true;
    accept();
}

void GrabCutDialogImpl::on_buttonBox_rejected()
{
    accepted_ = false;
    reject();
}

// ─── Commit result to GA[] ────────────────────────────────────────────────────

void GrabCutDialogImpl::commitResultToGA()
{
    if (segmentIndex_ < 0 || segmentIndex_ >= SegmentCount) return;
    if (!stateValid_) return;

    // Load GA image for this segment / slice
    LoadAllData(CurrentFile);

    QImage *gaImage = GA.at(segmentIndex_);
    if (!gaImage || gaImage->isNull()) return;

    QByteArray gaData = grabCut_.alphaAsGAImage(fwidth4);
    uchar *bits = gaImage->bits();
    for (int b = 0; b < gaData.size(); b++)
        bits[b] = static_cast<uchar>(gaData.at(b));

    // Save to disk
    SaveGreyData(CurrentFile, segmentIndex_);

    Segments[segmentIndex_]->Dirty   = true;
    if (CurrentFile < FilesDirty.size())
        FilesDirty[CurrentFile] = true;
}

// ─── Propagation ─────────────────────────────────────────────────────────────

void GrabCutDialogImpl::on_PropagateButton_clicked()
{
    if (!stateValid_)
    {
        QMessageBox::warning(this, "No seed",
                             "Run GrabCut on this slice first to create a seed segmentation.");
        return;
    }

    int fromSlice = PropFromSpinBox->value();
    int toSlice   = PropToSpinBox->value();

    if (fromSlice > toSlice)
    {
        QMessageBox::warning(this, "Invalid range",
                             "From slice must be ≤ To slice.");
        return;
    }

    // Commit seed slice first
    commitResultToGA();

    // Build the full trimap list (current slice + any previously painted trimaps)
    saveCurrentTrimap();

    // Set up propagation parameters
    PropagationParams params;
    params.seedSlice         = CurrentFile;
    params.firstSlice        = fromSlice;
    params.lastSlice         = toSlice;
    params.segmentIndex      = segmentIndex_;
    params.iterations        = PropIterationsSpinBox->value();
    params.overwriteExisting = PropOverwriteCheckBox->isChecked();
    params.confidenceThreshold = 0.95;

    // Create propagation object and move to thread
    propagation_ = new SlicePropagation();
    propagation_->setSeedState(currentState_);
    propagation_->setTrimaps(sliceTrimaps_);
    propagation_->setFileList(Files);
    propagation_->setImageDimensions(grabCut_.width(), grabCut_.height(), fwidth4);
    propagation_->setParams(params);

    propThread_ = new QThread(this);
    propagation_->moveToThread(propThread_);

    connect(propThread_,   &QThread::started,
            propagation_,  [this]() { propagation_->propagate(); });
    connect(propagation_,  &SlicePropagation::progressUpdated,
            this,          &GrabCutDialogImpl::onPropagationProgress,
            Qt::QueuedConnection);
    connect(propagation_,  &SlicePropagation::propagationComplete,
            this,          &GrabCutDialogImpl::onPropagationComplete,
            Qt::QueuedConnection);
    connect(propThread_,   &QThread::finished,
            propagation_,  &QObject::deleteLater);

    setPropagationRunning(true);
    propThread_->start();
}

void GrabCutDialogImpl::on_CancelPropButton_clicked()
{
    if (propagation_) propagation_->cancel();
    StatusLabel->setText("Cancelling...");
}

void GrabCutDialogImpl::onPropagationProgress(int percent, int sliceIdx)
{
    ProgressBar->setValue(percent);
    StatusLabel->setText(QString("Propagating... slice %1 (%2%)")
                         .arg(sliceIdx + 1).arg(percent));
}

void GrabCutDialogImpl::onPropagationComplete(std::vector<PropagationResult> results,
                                               bool cancelled)
{
    propThread_->quit();
    propThread_->wait();
    delete propThread_;
    propThread_  = nullptr;
    propagation_ = nullptr;  // already deleteLater'd

    setPropagationRunning(false);
    ProgressBar->setValue(cancelled ? ProgressBar->value() : 100);

    showPropagationSummary(results);

    // Reload current slice display
    LoadAllData(CurrentFile);
}

// ─── UI helpers ───────────────────────────────────────────────────────────────

void GrabCutDialogImpl::setPropagationRunning(bool running)
{
    PropagateButton->setEnabled(!running);
    CancelPropButton->setEnabled(running);
    RunButton->setEnabled(!running);
    RefineButton->setEnabled(!running);
    buttonBox->setEnabled(!running);
}

void GrabCutDialogImpl::showPropagationSummary(const std::vector<PropagationResult> &results)
{
    int total     = static_cast<int>(results.size());
    int succeeded = 0, needsReview = 0;
    for (const auto &r : results)
    {
        if (r.success) succeeded++;
        if (r.needsReview) needsReview++;
    }

    QString msg = QString("Propagation complete: %1/%2 slices processed.")
                  .arg(succeeded).arg(total);
    if (needsReview > 0)
        msg += QString("\n%1 slices flagged for review (low confidence) — check manually.").arg(needsReview);
    else
        msg += " All slices high-confidence.";

    StatusLabel->setText(msg);
    if (needsReview > 0)
        QMessageBox::information(this, "Propagation complete", msg);
}

void GrabCutDialogImpl::saveCurrentTrimap()
{
    sliceTrimaps_[static_cast<size_t>(CurrentFile)] = Canvas->trimap();
}
