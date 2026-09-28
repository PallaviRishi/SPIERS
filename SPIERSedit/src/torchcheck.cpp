/**
 * @file
 * Source: Torch Check
 *
 * Minimal Phase-0 sanity check that LibTorch links and runs inside SPIERSedit.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 */

// LibTorch headers must come before any Qt headers: Qt #defines keywords such
// as `slots`/`signals`/`emit` (and other short tokens) that collide with
// identifiers inside LibTorch's headers. Including torch first avoids this.
#include <torch/torch.h>

#include "torchcheck.h"

QString TorchCheck()
{
    torch::Tensor t = torch::rand({2, 3});
    float s = t.sum().item<float>();
    return QString("LibTorch OK (sum=%1)").arg(static_cast<double>(s), 0, 'f', 2);
}
