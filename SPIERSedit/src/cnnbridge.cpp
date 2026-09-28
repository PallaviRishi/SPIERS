/**
 * @file
 * Source: CNN Bridge
 *
 * Torch-including implementation of the torch-free RunCnnGenerate() entry
 * point. Keeps LibTorch headers out of Qt translation units.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 */

// torch first (before any Qt header) — via cnninterface.h.
#include "cnninterface.h"
#include "cnnbridge.h"

void RunCnnGenerate(QListWidget *sliceSelector)
{
    // One segmenter instance per run for now (retrains each generate, like the
    // other generation methods produce fresh output).
    static CnnInterface iface;
    iface.Generate(sliceSelector);
}
