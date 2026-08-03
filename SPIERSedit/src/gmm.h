/**
 * @file
 * Header: Gaussian Mixture Model
 *
 * Implements a K-component GMM in RGB colour space using the EM algorithm.
 * Used by GrabCut to model foreground and background colour distributions.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2024 by the SPIERS contributors.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 */

#ifndef GMM_H
#define GMM_H

#include <vector>
#include <array>
#include <cmath>
#include <limits>

/**
 * @brief A single Gaussian component in RGB space.
 *
 * Stores the mean (3D), full covariance matrix (3x3), its inverse and
 * determinant, and the mixture weight (pi).
 */
struct GMMComponent
{
    double mean[3];       ///< RGB mean
    double cov[3][3];     ///< Covariance matrix
    double covInv[3][3];  ///< Inverse covariance
    double covDet;        ///< Determinant of covariance
    double weight;        ///< Mixture weight (pi_k), sums to 1 across components
    int    sampleCount;   ///< Number of samples assigned in E-step (for M-step)

    GMMComponent();
    void reset();
};

/**
 * @brief K-component Gaussian Mixture Model for RGB colour data.
 *
 * Usage:
 *   1. Construct with desired number of components K.
 *   2. Call initFromSamples() with a set of RGB pixels.
 *   3. Repeatedly call eStep() then mStep() until convergence (or fixed iters).
 *   4. Use probability() to get per-pixel membership probability.
 *   5. Use mostLikelyComponent() for hard assignment (used in GrabCut E-step).
 */
class GMM
{
public:
    static constexpr int K = 5; ///< Number of mixture components

    GMM();

    /**
     * @brief Initialise components via k-means++ seeding on the given samples.
     * @param pixels  Flat list of RGB triples. Size must be 3*N.
     * @param N       Number of pixels.
     */
    void initFromSamples(const std::vector<double> &pixels, int N);

    /**
     * @brief E-step: assign each sample to the most likely component.
     *        Returns the component index for pixel i.
     */
    int mostLikelyComponent(double r, double g, double b) const;

    /**
     * @brief Compute the weighted log-likelihood (negative energy) for one pixel.
     *        Returns  sum_k  w_k * N(x | mu_k, Sigma_k)
     *        Used in GrabCut T-link computation.
     */
    double probability(double r, double g, double b) const;

    /**
     * @brief Full EM iteration: E-step + M-step on the provided sample set.
     * @param pixels  Flat RGB triples, size 3*N.
     * @param N       Number of pixels.
     * @param iters   Number of EM iterations to run.
     */
    void fit(const std::vector<double> &pixels, int N, int iters = 10);

    /**
     * @brief Accumulate one sample into component k (used in GrabCut's per-pixel
     *        component assignment step before calling learnFromAccumulated()).
     */
    void addSample(int k, double r, double g, double b);

    /**
     * @brief Recompute component parameters from accumulated samples.
     *        Call after addSample() loops to complete the M-step.
     */
    void learnFromAccumulated();

    /**
     * @brief Reset all accumulators (call before a new E-step accumulation pass).
     */
    void resetAccumulators();

    // Serialisation helpers (for .spe persistence)
    void serialise(std::vector<double> &out) const;
    void deserialise(const std::vector<double> &in);
    static int serialisedSize() { return K * (3 + 9 + 1); } // mean + cov + weight

    GMMComponent components[K];

private:
    // Accumulators for M-step
    double accumMean[K][3];
    double accumCov[K][3][3];
    double accumWeight[K];
    int    accumCount[K];

    void computeInverseAndDet(GMMComponent &c);
    double gaussianPDF(const GMMComponent &c, double r, double g, double b) const;

    // k-means++ initialisation
    void kMeansInit(const std::vector<double> &pixels, int N);
    void runKMeans(const std::vector<double> &pixels, int N, int iters);
};

#endif // GMM_H
