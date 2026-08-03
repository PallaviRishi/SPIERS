/**
 * @file
 * Header: GrabCut State Persistence
 *
 * Handles saving and loading of GrabCut-specific state to the .spe settings
 * folder. This includes per-slice trimap scribbles, per-segment GMM parameters,
 * and the grabCutAlphaCache entries.
 *
 * File naming convention (mirrors existing SPIERSedit pattern):
 *   Trimaps: {settingsDir}/gc_trimap_s{seg+1}_{imagename}.dat
 *   GMM:     {settingsDir}/gc_gmm_s{seg+1}_{imagename}.dat
 *   Alpha:   {settingsDir}/gc_alpha_s{seg+1}_{imagename}.dat
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2026 by the SPIERS contributors.
 */

#ifndef GRABCUTPERSISTENCE_H
#define GRABCUTPERSISTENCE_H

#include "gmm.h"
#include "grabcut.h"

#include <QByteArray>
#include <QString>

#include <vector>

/**
 * @brief Save a trimap for the given segment and slice to the settings folder.
 * @param segmentIndex  Segment index (0-based).
 * @param fileIndex     Slice/file index (0-based).
 * @param trimapData    Flat trimap byte array (one byte per pixel, row-major).
 */
void saveGrabCutTrimap(int segmentIndex, int fileIndex, const QByteArray &trimapData);

/**
 * @brief Load a previously saved trimap for the given segment and slice.
 * @param segmentIndex  Segment index (0-based).
 * @param fileIndex     Slice/file index (0-based).
 * @return  The trimap byte array, or an empty QByteArray if none was found.
 */
QByteArray loadGrabCutTrimap(int segmentIndex, int fileIndex);

/**
 * @brief Save a GrabCutState (fg/bg GMMs) for the given segment and slice.
 * @param segmentIndex  Segment index (0-based).
 * @param fileIndex     Slice/file index (0-based).
 * @param state         The GrabCut state to persist (only GMMs are serialised).
 */
void saveGrabCutGmmState(int segmentIndex, int fileIndex, const GrabCutState &state);

/**
 * @brief Load a previously saved GrabCutState for the given segment and slice.
 * @param segmentIndex  Segment index (0-based).
 * @param fileIndex     Slice/file index (0-based).
 * @param state         Output: the loaded state. isInitialised set to true on success.
 * @return  True if a saved state was found and loaded, false otherwise.
 */
bool loadGrabCutGmmState(int segmentIndex, int fileIndex, GrabCutState &state);

/**
 * @brief Save an alpha cache entry for the given segment and slice.
 * @param segmentIndex  Segment index (0-based).
 * @param fileIndex     Slice/file index (0-based).
 * @param alphaData     Alpha byte array in GA[] stride format.
 */
void saveGrabCutAlpha(int segmentIndex, int fileIndex, const QByteArray &alphaData);

/**
 * @brief Load a previously saved alpha cache entry.
 * @param segmentIndex  Segment index (0-based).
 * @param fileIndex     Slice/file index (0-based).
 * @return  The alpha byte array, or an empty QByteArray if none was found.
 */
QByteArray loadGrabCutAlpha(int segmentIndex, int fileIndex);

/**
 * @brief Load all saved GrabCut alpha data into grabCutAlphaCache on project open.
 *        Scans the settings folder for gc_alpha_* files and populates the cache.
 */
void loadAllGrabCutAlphaIntoCache();

/**
 * @brief Save all current grabCutAlphaCache entries to disk.
 *        Called during project save.
 */
void saveAllGrabCutAlphaFromCache();

#endif // GRABCUTPERSISTENCE_H
