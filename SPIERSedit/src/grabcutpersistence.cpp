/**
 * @file
 * Source: GrabCut State Persistence
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2026 by the SPIERS contributors.
 */

#include "grabcutpersistence.h"
#include "globals.h"
#include "grabcutglobals.h"

#include <QDir>
#include <QFile>
#include <QDataStream>
#include <QIODevice>

#include <cstring>

// ─── Path helpers ─────────────────────────────────────────────────────────────

/**
 * @brief Build the settings directory path for the current dataset.
 *        Mirrors the pattern used by SaveMasks/SaveGreyData in fileio.cpp.
 * @param fileIndex  Slice/file index.
 * @return  Absolute path to the settings directory.
 */
static QString settingsDir(int fileIndex)
{
    QString fname = Files.at(fileIndex);
    int lastSep = qMax(fname.lastIndexOf("\\"), fname.lastIndexOf("/"));
    QString dir = fname.left(lastSep);
    dir.append("/" + SettingsFileName);
    return dir;
}

/**
 * @brief Extract the base image name (without extension) for a given slice.
 * @param fileIndex  Slice/file index.
 * @return  Image base name (e.g. "fossil_001").
 */
static QString imageBaseName(int fileIndex)
{
    QString fname = Files.at(fileIndex);
    int lastSep = qMax(fname.lastIndexOf("\\"), fname.lastIndexOf("/"));
    int lastDot = fname.lastIndexOf(".");
    return fname.mid(lastSep + 1, lastDot - lastSep - 1);
}

/**
 * @brief Build a full file path for a GrabCut persistence file.
 * @param fileIndex     Slice/file index.
 * @param segmentIndex  Segment index.
 * @param prefix        File prefix ("gc_trimap", "gc_gmm", or "gc_alpha").
 * @return  Full absolute path including extension.
 */
static QString buildPath(int fileIndex, int segmentIndex, const QString &prefix)
{
    QString dir = settingsDir(fileIndex);
    QString baseName = imageBaseName(fileIndex);
    QString segStr = QString("s%1_").arg(segmentIndex + 1);
    return dir + "/" + prefix + "_" + segStr + baseName + ".dat";
}

// ─── Trimap persistence ───────────────────────────────────────────────────────

void saveGrabCutTrimap(int segmentIndex, int fileIndex, const QByteArray &trimapData)
{
    if (fileIndex < 0 || fileIndex >= FileCount) return;
    if (trimapData.isEmpty()) return;

    QString path = buildPath(fileIndex, segmentIndex, "gc_trimap");

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly))
        return;

    file.write(trimapData);
    file.close();
}

QByteArray loadGrabCutTrimap(int segmentIndex, int fileIndex)
{
    if (fileIndex < 0 || fileIndex >= FileCount)
        return QByteArray();

    QString path = buildPath(fileIndex, segmentIndex, "gc_trimap");

    QFile file(path);
    if (!file.exists() || !file.open(QIODevice::ReadOnly))
        return QByteArray();

    QByteArray data = file.readAll();
    file.close();
    return data;
}

// ─── GMM state persistence ────────────────────────────────────────────────────

void saveGrabCutGmmState(int segmentIndex, int fileIndex, const GrabCutState &state)
{
    if (fileIndex < 0 || fileIndex >= FileCount) return;

    QString path = buildPath(fileIndex, segmentIndex, "gc_gmm");

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly))
        return;

    QDataStream stream(&file);
    stream.setVersion(QDataStream::Qt_5_0);

    // Serialise both GMMs as flat double vectors
    std::vector<double> fgData;
    std::vector<double> bgData;
    state.fgGmm.serialise(fgData);
    state.bgGmm.serialise(bgData);

    // Write fg GMM
    stream << static_cast<qint32>(fgData.size());
    for (double val : fgData)
        stream << val;

    // Write bg GMM
    stream << static_cast<qint32>(bgData.size());
    for (double val : bgData)
        stream << val;

    file.close();
}

bool loadGrabCutGmmState(int segmentIndex, int fileIndex, GrabCutState &state)
{
    if (fileIndex < 0 || fileIndex >= FileCount)
        return false;

    QString path = buildPath(fileIndex, segmentIndex, "gc_gmm");

    QFile file(path);
    if (!file.exists() || !file.open(QIODevice::ReadOnly))
        return false;

    QDataStream stream(&file);
    stream.setVersion(QDataStream::Qt_5_0);

    // Read fg GMM
    qint32 fgSize = 0;
    stream >> fgSize;
    if (fgSize != GMM::serialisedSize())
    {
        file.close();
        return false;
    }
    std::vector<double> fgData(static_cast<size_t>(fgSize));
    for (int i = 0; i < fgSize; i++)
        stream >> fgData[static_cast<size_t>(i)];

    // Read bg GMM
    qint32 bgSize = 0;
    stream >> bgSize;
    if (bgSize != GMM::serialisedSize())
    {
        file.close();
        return false;
    }
    std::vector<double> bgData(static_cast<size_t>(bgSize));
    for (int i = 0; i < bgSize; i++)
        stream >> bgData[static_cast<size_t>(i)];

    file.close();

    state.fgGmm.deserialise(fgData);
    state.bgGmm.deserialise(bgData);
    state.isInitialised = true;

    return true;
}

// ─── Alpha persistence ────────────────────────────────────────────────────────

void saveGrabCutAlpha(int segmentIndex, int fileIndex, const QByteArray &alphaData)
{
    if (fileIndex < 0 || fileIndex >= FileCount) return;
    if (alphaData.isEmpty()) return;

    QString path = buildPath(fileIndex, segmentIndex, "gc_alpha");

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly))
        return;

    file.write(alphaData);
    file.close();
}

QByteArray loadGrabCutAlpha(int segmentIndex, int fileIndex)
{
    if (fileIndex < 0 || fileIndex >= FileCount)
        return QByteArray();

    QString path = buildPath(fileIndex, segmentIndex, "gc_alpha");

    QFile file(path);
    if (!file.exists() || !file.open(QIODevice::ReadOnly))
        return QByteArray();

    QByteArray data = file.readAll();
    file.close();
    return data;
}

// ─── Bulk cache operations ────────────────────────────────────────────────────

void loadAllGrabCutAlphaIntoCache()
{
    grabCutAlphaCache.clear();

    for (int seg = 0; seg < SegmentCount; seg++)
    {
        for (int file = 0; file < FileCount; file++)
        {
            QByteArray alpha = loadGrabCutAlpha(seg, file);
            if (!alpha.isEmpty())
                grabCutAlphaCache[QPair<int, int>(seg, file)] = alpha;
        }
    }
}

void saveAllGrabCutAlphaFromCache()
{
    QMapIterator<QPair<int, int>, QByteArray> it(grabCutAlphaCache);
    while (it.hasNext())
    {
        it.next();
        int seg = it.key().first;
        int file = it.key().second;
        saveGrabCutAlpha(seg, file, it.value());
    }
}
