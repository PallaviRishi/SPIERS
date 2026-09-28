/**
 * @file
 * Header: CNN Bridge
 *
 * A torch-free entry point so Qt translation units (e.g. mainwindow2.cpp) can
 * trigger CNN segmentation without including any LibTorch headers (which would
 * clash with Qt's keyword macros). The implementation lives in cnnbridge.cpp,
 * which includes torch before Qt.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 */

#ifndef CNNBRIDGE_H
#define CNNBRIDGE_H

class QListWidget;

/**
 * @brief Run the CNN segmentation generate pipeline on the selected slices.
 *        Safe to call from Qt-only code — pulls in no torch headers here.
 */
void RunCnnGenerate(QListWidget *sliceSelector);

#endif // CNNBRIDGE_H
