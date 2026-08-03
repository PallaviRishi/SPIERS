/**
 * @file
 * Header: GrabCut Global State
 *
 * Declares global variables that are specific to the GrabCut segmentation
 * system. Kept in a separate header so that the main globals.h does not
 * acquire Qt container includes.
 *
 * Include this header only in files that directly use the GrabCut alpha cache
 * (grabcutdialogimpl.cpp, display.cpp generation pipeline).
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2026 by the SPIERS contributors.
 */

#ifndef GRABCUTGLOBALS_H
#define GRABCUTGLOBALS_H

#include <QByteArray>
#include <QMap>
#include <QPair>

/**
 * @brief Per-segment, per-slice GrabCut alpha cache.
 *
 * Key:   QPair<int, int>(segmentIndex, sliceIndex)
 * Value: Flat alpha byte array in SPIERSedit GA[] stride format (255 = fossil, 0 = background).
 *
 * Populated by GrabCutDialogImpl when the user accepts a segmentation result
 * or when slice propagation writes a result. Consumed by MakeGrabCutGreyScale()
 * in the generation pipeline so a saved GrabCut result can be re-applied when
 * the user regenerates a segment.
 */
extern QMap<QPair<int, int>, QByteArray> grabCutAlphaCache;

#endif // GRABCUTGLOBALS_H
