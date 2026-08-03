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
 * Copyright 2026 by the SPIERS contributors.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 */

#ifndef GMM_H
#define GMM_H

#include <cmath>
#include <limits>
#include <vector>

/**
 * @brief A single Gaussian component in RGB space.
 *
 * Stores the mean (3D), full covariance matrix (3x3), its inverse and
 * determinant, and the mixture weight (pi).
 */
struct GMMComponent
{
    double cov[3][3];     ///< Covariance matrix
    double covDet;        ///< Determinant of covariance
    double covInv[3][3];  ///< Inverse covariance
    double mean[3];       ///< RGB mean
    double weight;        ///< Mixture weight (pi_k), sums to 1 across components
    int sampleCount;      ///< Number of samples assigned in E-step (for M-step)

    GMMComponent();
    void reset();
};

/**
 * @brief K-component Gaussian Mixture Model for RGB colour data.
 *
 * Usage:
 *   1. Construct the GMM.
 *   2. Call initFromSamples() with a set of RGB pixels.
 *   3. Call fit() to run EM iterations.
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
     * @brief Return the index of the component most likely to have generated (r,g,b).
     * @param r  Red channel value (0–1).
     * @param g  Green channel value (0–1).
     * @param b  Blue channel value (0–1).
     * @return   Component index in range 0..K-1.
     */
    int mostLikelyComponent(double r, double g, double b) const;

    /**
     * @brief Compute the mixture probability density at (r,g,b).
     *        Returns sum_k w_k * N(x | mu_k, Sigma_k).
     *        Used in GrabCut T-link computation.
     * @param r  Red channel value (0–1).
     * @param g  Green channel value (0–1).
     * @param b  Blue channel value (0–1).
     * @return   Mixture probability density.
     */
    double probability(double r, double g, double b) const;

    /**
     * @brief Compute the non-negative data energy for pixel (r,g,b) assigned to
     *        component k. Correct formulation for graph cut T-links:
     *        E = -log(pi_k) + 0.5*log(det(Sigma_k)) + 0.5*mahalanobis^2
     *        Always >= 0.
     * @param k  Component index (0..K-1).
     * @param r  Red channel value (0-1).
     * @param g  Green channel value (0-1).
     * @param b  Blue channel value (0-1).
     * @return   Non-negative energy value.
     */
    double componentEnergy(int k, double r, double g, double b) const;
    /**
     * @brief Full EM iteration: E-step + M-step on the provided sample set.
     * @param pixels  Flat RGB triples, size 3*N.
     * @param N       Number of pixels.
     * @param iters   Number of EM iterations to run.
     */
    void fit(const std::vector<double> &pixels, int N, int iters = 10);

    /**
     * @brief Accumulate one sample into component k.
     *        Used in GrabCut's per-pixel component assignment step before
     *        calling learnFromAccumulated().
     * @param k  Component index.
     * @param r  Red channel value (0–1).
     * @param g  Green channel value (0–1).
     * @param b  Blue channel value (0–1).
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

    /**
     * @brief Serialise all component parameters into a flat double vector.
     * @param out  Output vector; values are appended.
     */
    void serialise(std::vector<double> &out) const;

    /**
     * @brief Restore component parameters from a flat double vector produced
     *        by serialise().
     * @param in  Input vector.
     */
    void deserialise(const std::vector<double> &in);

    /**
     * @brief Number of doubles written/read by serialise() / deserialise().
     * @return  K * (weight + 3 means + 9 covariance values).
     */
    static int serialisedSize() { return K * (1 + 3 + 9); }

    GMMComponent components[K]; ///< Public so GrabCut can access component weights

private:
    double accumCov[K][3][3];   ///< Covariance accumulators for M-step
    int accumCount[K];          ///< Sample count accumulators for M-step
    double accumMean[K][3];     ///< Mean accumulators for M-step
    double accumWeight[K];      ///< Weight accumulators for M-step

    /**
     * @brief Compute and cache the inverse and determinant of c.cov.
     *        Called after each M-step update.
     * @param c  Component to update in-place.
     */
    void computeInverseAndDet(GMMComponent &c);

    /**
     * @brief Evaluate the multivariate Gaussian PDF for component c at (r,g,b).
     * @param c  Gaussian component.
     * @param r  Red channel value (0–1).
     * @param g  Green channel value (0–1).
     * @param b  Blue channel value (0–1).
     * @return   PDF value.
     */
    double gaussianPDF(const GMMComponent &c, double r, double g, double b) const;

    /**
     * @brief Seed component means using the k-means++ distance-proportional scheme.
     * @param pixels  Flat RGB triples.
     * @param N       Number of pixels.
     */
    void kMeansInit(const std::vector<double> &pixels, int N);

    /**
     * @brief Run k-means hard assignment and mean update for a fixed number of
     *        iterations to improve the initial seeding before EM.
     * @param pixels  Flat RGB triples.
     * @param N       Number of pixels.
     * @param iters   Number of k-means iterations.
     */
    void runKMeans(const std::vector<double> &pixels, int N, int iters);
};

#endif // GMM_H
