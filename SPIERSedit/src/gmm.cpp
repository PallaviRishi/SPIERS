/**
 * @file
 * Source: Gaussian Mixture Model
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2026 by the SPIERS contributors.
 */

#include "gmm.h"
#include <cstring>
#include <cassert>
#include <algorithm>
#include <random>
#include <numeric>

// ─── GMMComponent ────────────────────────────────────────────────────────────

GMMComponent::GMMComponent()
{
    reset();
}

void GMMComponent::reset()
{
    std::fill(std::begin(mean), std::end(mean), 0.0);
    for (int i = 0; i < 3; i++)
        std::fill(std::begin(cov[i]), std::end(cov[i]), 0.0);
    for (int i = 0; i < 3; i++)
        std::fill(std::begin(covInv[i]), std::end(covInv[i]), 0.0);
    covDet     = 1.0;
    weight     = 1.0 / GMM::K;
    sampleCount = 0;
}

// ─── GMM private helpers ─────────────────────────────────────────────────────

/**
 * @brief Compute the 3x3 matrix inverse and determinant for a GMMComponent.
 *
 * Uses the analytic formula for 3x3 symmetric positive-definite matrices.
 * If the matrix is near-singular we add a small ridge (regularisation) to
 * prevent numerical blowup with very small or degenerate clusters.
 */
void GMM::computeInverseAndDet(GMMComponent &c)
{
    // Regularise diagonal slightly to avoid singularity
    const double ridge = 1e-6;
    double m[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            m[i][j] = c.cov[i][j] + (i == j ? ridge : 0.0);

    // Determinant (cofactor expansion along row 0)
    double det =
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if (std::fabs(det) < 1e-20) det = 1e-20; // failsafe
    c.covDet = det;

    // Adjugate / det
    c.covInv[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
    c.covInv[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) / det;
    c.covInv[0][2] =  (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
    c.covInv[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) / det;
    c.covInv[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
    c.covInv[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) / det;
    c.covInv[2][0] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
    c.covInv[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) / det;
    c.covInv[2][2] =  (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
}

/**
 * @brief Evaluate the multivariate Gaussian PDF for component c at (r,g,b).
 */
double GMM::gaussianPDF(const GMMComponent &c, double r, double g, double b) const
{
    double d[3] = { r - c.mean[0], g - c.mean[1], b - c.mean[2] };

    // Mahalanobis squared distance:  d^T * Sigma^{-1} * d
    double mah = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            mah += d[i] * c.covInv[i][j] * d[j];

    double det = std::fabs(c.covDet);
    if (det < 1e-30) det = 1e-30;

    // (2*pi)^(3/2) ≈ 15.7496
    double norm = 1.0 / (15.7496 * std::sqrt(det));
    return norm * std::exp(-0.5 * mah);
}

// ─── k-means++ initialisation ────────────────────────────────────────────────

void GMM::kMeansInit(const std::vector<double> &pixels, int N)
{
    if (N == 0) return;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> uniformPick(0, N - 1);

    // Choose first centre uniformly at random
    std::vector<int> chosen;
    chosen.push_back(uniformPick(rng));

    // k-means++ distance-proportional seeding
    std::vector<double> dist2(static_cast<size_t>(N), std::numeric_limits<double>::max());

    for (int k = 1; k < K; k++)
    {
        int prev = chosen.back();
        double pr = pixels[static_cast<size_t>(prev) * 3 + 0];
        double pg = pixels[static_cast<size_t>(prev) * 3 + 1];
        double pb = pixels[static_cast<size_t>(prev) * 3 + 2];

        double sum = 0.0;
        for (int i = 0; i < N; i++)
        {
            double dr = pixels[static_cast<size_t>(i) * 3 + 0] - pr;
            double dg = pixels[static_cast<size_t>(i) * 3 + 1] - pg;
            double db = pixels[static_cast<size_t>(i) * 3 + 2] - pb;
            double d  = dr * dr + dg * dg + db * db;
            if (d < dist2[static_cast<size_t>(i)]) dist2[static_cast<size_t>(i)] = d;
            sum += dist2[static_cast<size_t>(i)];
        }

        // Sample proportional to dist2
        std::uniform_real_distribution<double> ud(0.0, sum);
        double target = ud(rng);
        double acc = 0.0;
        int    pick = 0;
        for (int i = 0; i < N; i++)
        {
            acc += dist2[static_cast<size_t>(i)];
            if (acc >= target) { pick = i; break; }
        }
        chosen.push_back(pick);
    }

    // Set means from chosen seeds, identity covariance, equal weights
    for (int k = 0; k < K; k++)
    {
        int seedIdx = chosen[static_cast<size_t>(k)];
        components[k].mean[0] = pixels[static_cast<size_t>(seedIdx) * 3 + 0];
        components[k].mean[1] = pixels[static_cast<size_t>(seedIdx) * 3 + 1];
        components[k].mean[2] = pixels[static_cast<size_t>(seedIdx) * 3 + 2];

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                components[k].cov[i][j] = (i == j) ? 1000.0 : 0.0; // broad initial cov

        components[k].weight = 1.0 / K;
        computeInverseAndDet(components[k]);
    }
}

void GMM::runKMeans(const std::vector<double> &pixels, int N, int iters)
{
    std::vector<int> assignments(static_cast<size_t>(N), 0);

    for (int iter = 0; iter < iters; iter++)
    {
        // Assignment step
        for (int i = 0; i < N; i++)
        {
            double best = std::numeric_limits<double>::max();
            int    bestK = 0;
            double r = pixels[static_cast<size_t>(i) * 3 + 0];
            double g = pixels[static_cast<size_t>(i) * 3 + 1];
            double b = pixels[static_cast<size_t>(i) * 3 + 2];
            for (int k = 0; k < K; k++)
            {
                double dr = r - components[k].mean[0];
                double dg = g - components[k].mean[1];
                double db = b - components[k].mean[2];
                double d  = dr * dr + dg * dg + db * db;
                if (d < best) { best = d; bestK = k; }
            }
            assignments[static_cast<size_t>(i)] = bestK;
        }

        // Update means
        double newMean[K][3] = {};
        int    counts[K]     = {};
        for (int i = 0; i < N; i++)
        {
            int k = assignments[static_cast<size_t>(i)];
            newMean[k][0] += pixels[static_cast<size_t>(i) * 3 + 0];
            newMean[k][1] += pixels[static_cast<size_t>(i) * 3 + 1];
            newMean[k][2] += pixels[static_cast<size_t>(i) * 3 + 2];
            counts[k]++;
        }
        for (int k = 0; k < K; k++)
            if (counts[k] > 0)
                for (int d = 0; d < 3; d++)
                    components[k].mean[d] = newMean[k][d] / counts[k];
    }
}

// ─── GMM public interface ─────────────────────────────────────────────────────

GMM::GMM()
{
    resetAccumulators();
}

void GMM::initFromSamples(const std::vector<double> &pixels, int N)
{
    if (N < K)
    {
        // Not enough samples — fall back to simple uniform initialisation
        for (int k = 0; k < K; k++)
        {
            components[k].reset();
            components[k].weight = 1.0 / K;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    components[k].cov[i][j] = (i == j) ? 1000.0 : 0.0;
            computeInverseAndDet(components[k]);
        }
        return;
    }

    kMeansInit(pixels, N);
    runKMeans(pixels, N, 10);

    // Compute covariances and weights directly from hard k-means assignments.
    // Soft EM with broad initial covariances collapses all components to the
    // global mean, so we compute the cluster statistics explicitly.
    std::vector<int> assignments(static_cast<size_t>(N), 0);
    for (int i = 0; i < N; i++)
    {
        double best = std::numeric_limits<double>::max();
        int bestK = 0;
        double r = pixels[static_cast<size_t>(i) * 3 + 0];
        double g = pixels[static_cast<size_t>(i) * 3 + 1];
        double b = pixels[static_cast<size_t>(i) * 3 + 2];
        for (int k = 0; k < K; k++)
        {
            double dr = r - components[k].mean[0];
            double dg = g - components[k].mean[1];
            double db = b - components[k].mean[2];
            double d = dr * dr + dg * dg + db * db;
            if (d < best) { best = d; bestK = k; }
        }
        assignments[static_cast<size_t>(i)] = bestK;
    }
    for (int k = 0; k < K; k++)
    {
        int count = 0;
        double covAccum[3][3] = {};
        for (int i = 0; i < N; i++)
        {
            if (assignments[static_cast<size_t>(i)] != k) continue;
            count++;
            double d[3] = {
                pixels[static_cast<size_t>(i) * 3 + 0] - components[k].mean[0],
                pixels[static_cast<size_t>(i) * 3 + 1] - components[k].mean[1],
                pixels[static_cast<size_t>(i) * 3 + 2] - components[k].mean[2]
            };
            for (int p = 0; p < 3; p++)
                for (int q = 0; q < 3; q++)
                    covAccum[p][q] += d[p] * d[q];
        }
        if (count > 1)
        {
            components[k].weight = static_cast<double>(count) / N;
            for (int p = 0; p < 3; p++)
                for (int q = 0; q < 3; q++)
                    components[k].cov[p][q] = covAccum[p][q] / count;
        }
        else
        {
            components[k].weight = 1.0 / K;
            for (int p = 0; p < 3; p++)
                for (int q = 0; q < 3; q++)
                    components[k].cov[p][q] = (p == q) ? 0.01 : 0.0;
        }
        computeInverseAndDet(components[k]);
    }
}

int GMM::mostLikelyComponent(double r, double g, double b) const
{
    double best = -1.0;
    int    bestK = 0;
    for (int k = 0; k < K; k++)
    {
        double p = components[k].weight * gaussianPDF(components[k], r, g, b);
        if (p > best) { best = p; bestK = k; }
    }
    return bestK;
}

double GMM::probability(double r, double g, double b) const
{
    double p = 0.0;
    for (int k = 0; k < K; k++)
        p += components[k].weight * gaussianPDF(components[k], r, g, b);
    return p;
}

void GMM::fit(const std::vector<double> &pixels, int N, int iters)
{
    for (int iter = 0; iter < iters; iter++)
    {
        // ── E-step: compute responsibilities ────────────────────────────────
        // We do a "hard" assignment (Viterbi EM) for stability with small N,
        // falling back to soft assignment when N is large.
        // For GrabCut's usage hard assignment is standard (Rother et al. 2004).

        resetAccumulators();

        for (int i = 0; i < N; i++)
        {
            double r = pixels[static_cast<size_t>(i) * 3 + 0];
            double g = pixels[static_cast<size_t>(i) * 3 + 1];
            double b = pixels[static_cast<size_t>(i) * 3 + 2];

            // Compute posterior for each component
            double resp[K];
            double sum = 0.0;
            for (int k = 0; k < K; k++)
            {
                resp[k] = components[k].weight * gaussianPDF(components[k], r, g, b);
                sum += resp[k];
            }
            if (sum < 1e-300) sum = 1e-300;
            for (int k = 0; k < K; k++) resp[k] /= sum;

            // Accumulate
            for (int k = 0; k < K; k++)
            {
                accumWeight[k] += resp[k];
                accumMean[k][0] += resp[k] * r;
                accumMean[k][1] += resp[k] * g;
                accumMean[k][2] += resp[k] * b;
                for (int p = 0; p < 3; p++)
                {
                    double dp[3] = { r, g, b };
                    for (int q = 0; q < 3; q++)
                        accumCov[k][p][q] += resp[k] * dp[p] * dp[q];
                }
                accumCount[k]++;
            }
        }

        learnFromAccumulated();
    }
}

void GMM::resetAccumulators()
{
    for (int k = 0; k < K; k++)
    {
        accumWeight[k] = 0.0;
        accumCount[k]  = 0;
        for (int i = 0; i < 3; i++)
        {
            accumMean[k][i] = 0.0;
            for (int j = 0; j < 3; j++)
                accumCov[k][i][j] = 0.0;
        }
    }
}

void GMM::addSample(int k, double r, double g, double b)
{
    accumWeight[k] += 1.0;
    accumCount[k]  += 1;
    accumMean[k][0] += r;
    accumMean[k][1] += g;
    accumMean[k][2] += b;

    double vals[3] = { r, g, b };
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            accumCov[k][i][j] += vals[i] * vals[j];
}

void GMM::learnFromAccumulated()
{
    double totalWeight = 0.0;
    for (int k = 0; k < K; k++) totalWeight += accumWeight[k];
    if (totalWeight < 1e-10) return;

    for (int k = 0; k < K; k++)
    {
        double wk = accumWeight[k];
        if (wk < 1e-10)
        {
            // Empty component — keep old parameters, just zero weight
            components[k].weight = 0.0;
            continue;
        }

        components[k].weight = wk / totalWeight;

        // Mean
        for (int i = 0; i < 3; i++)
            components[k].mean[i] = accumMean[k][i] / wk;

        // Covariance:  E[xx^T] - mu*mu^T
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                components[k].cov[i][j] =
                    accumCov[k][i][j] / wk
                    - components[k].mean[i] * components[k].mean[j];

        computeInverseAndDet(components[k]);
    }
}

// ─── Serialisation ────────────────────────────────────────────────────────────

void GMM::serialise(std::vector<double> &out) const
{
    for (int k = 0; k < K; k++)
    {
        const GMMComponent &c = components[k];
        out.push_back(c.weight);
        for (int i = 0; i < 3; i++) out.push_back(c.mean[i]);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                out.push_back(c.cov[i][j]);
    }
}

void GMM::deserialise(const std::vector<double> &in)
{
    size_t pos = 0;
    for (int k = 0; k < K; k++)
    {
        GMMComponent &c = components[k];
        c.weight = in[pos++];
        for (int i = 0; i < 3; i++) c.mean[i] = in[pos++];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                c.cov[i][j] = in[pos++];
        computeInverseAndDet(c);
    }
}
