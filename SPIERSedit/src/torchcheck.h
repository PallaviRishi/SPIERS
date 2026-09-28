/**
 * @file
 * Header: Torch Check
 *
 * Minimal Phase-0 sanity check that LibTorch links and runs inside SPIERSedit.
 * Temporary scaffolding for the CNN segmentation work; can be removed once the
 * real model code is in place.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 */

#ifndef TORCHCHECK_H
#define TORCHCHECK_H

#include <QString>

/**
 * @brief Run a trivial LibTorch tensor op and return a human-readable result.
 *        Used at startup to confirm the LibTorch toolchain is wired in.
 * @return A status string, e.g. "LibTorch OK (sum=2.35)".
 */
QString TorchCheck();

#endif // TORCHCHECK_H
