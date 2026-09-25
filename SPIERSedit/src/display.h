/**
 * @file
 * Header: Display
 *
 * All SPIERS code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * All SPIERS code is Copyright 2008-2026 by Mark D. Sutton, Russell J. Garwood,
 * and Alan R.T. Spencer.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version. This program is distributed in the
 * hope that it will be useful, but WITHOUT ANY WARRANTY.
 */

#ifndef __DISPLAY_H__
#define __DISPLAY_H__

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QString>
#include "globals.h"
#include "bytearray2d.h"
#include "beamhardening.h"
#include "gmm.h"
#include <QList>
#include "labelledpoint.h"

class QListWidget;

extern void ShowImage(QGraphicsView *gv);
extern void InitImage(QGraphicsView *gv);
extern void DeleteDisplayObjects();
extern QList<LabelledPoint> GenerateLabels(MainWindow *mw, int percentage);
extern void ClearImages();
extern void MakeLinearGreyScale(int seg, int fnum, bool flag);
extern void ApplyLCE(int seg, int fnum, bool flag);
extern void ApplyGradient(int seg, int fnum);
extern void ApplyRadial(int seg, int fnum, BeamHardening *bh, bool flag);
extern void MakeBlankGreyScale(int seg, int fnum, bool flag);
extern uchar GreyScalePixel(int w, int h, int r, int g, int b, int glob);
extern uchar LCEPixel(int w, int h, uchar *original_data, QByteArray *new_locks);
extern uchar RadialPixel(int w, int h, uchar *original_data, QByteArray *new_locks, BeamHardening *bh);
extern void MakePolyGreyScale(int seg, int fnum, bool flag);
extern void MakeRangeGreyScale(int seg, int fnum, bool flag);
extern uchar PolyPixel(int w, int h, int seg);
extern uchar RangePixel(int w, int h, int bot, int top, double cen, double gra, int seg);
extern uchar GenPixel(int x, int y, int s, QVector<uchar> *sample, QByteArray *locks);
extern double CalcPoly(unsigned char r, unsigned char g, unsigned char b, Segment *seg);
extern void SaveMainImage(QString fname);
extern QByteArray DoMaskLocking();
extern QVector<int> GetSegmentMap();

// ── GrabCut generation mode ───────────────────────────────────────────────────
/**
 * @brief Apply a pre-computed GrabCut alpha result to GA[seg] for file fnum.
 *
 * Called from the generation pipeline when a segment has a saved GrabCut state.
 * Loads the stored alpha QByteArray (one byte per pixel, 255=fg, 0=bg) and
 * writes it into GA[seg], respecting the mask-locking system.
 *
 * @param seg    Segment index
 * @param fnum   File (slice) index
 * @param flag   If true, skip LoadAllData / SaveGreyData (caller manages I/O)
 */
extern void MakeGmmGreyScale(int seg, int fnum, bool flag);

/**
 * @brief Train shared fg/bg GMMs from locked pixels across all selected slices.
 * @return true if both classes had enough samples to train.
 */
extern bool TrainGmmModels(QListWidget *SliceSelectorList, GMM &fgGmm, GMM &bgGmm, int fgSeg);

/**
 * @brief Classify all unlocked pixels of one slice using pre-trained GMMs,
 *        writing the result into segment seg. Applies to any slice, locked or not.
 */
extern void ClassifyGmmSlice(int fnum, const GMM &fgGmm, const GMM &bgGmm, int seg);

#endif // __DISPLAY_H__
