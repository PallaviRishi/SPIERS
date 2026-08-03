/**
 * @file
 * Header: Max-flow / Min-cut Graph Engine
 *
 * Implements Boykov-Kolmogorov push-relabel max-flow for graph cut segmentation.
 * Based on the algorithm described in:
 *   Boykov & Kolmogorov, "An Experimental Comparison of Min-Cut/Max-Flow
 *   Algorithms for Energy Minimization in Vision", PAMI 2004.
 *
 * The graph has:
 *   - One node per image pixel  (indices 0 .. N-1)
 *   - One source node  (index N)
 *   - One sink node    (index N+1)
 *
 * T-links  : pixel ↔ source  and  pixel ↔ sink  (unary/data terms)
 * N-links  : pixel ↔ pixel               (pairwise/smoothness terms)
 *
 * All edge capacities are non-negative doubles.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2024 by the SPIERS contributors.
 */

#ifndef MAXFLOW_H
#define MAXFLOW_H

#include <vector>
#include <cstdint>
#include <limits>

/**
 * @brief One directed edge in the residual graph.
 */
struct Edge
{
    int    dst;       ///< Destination node index
    double cap;       ///< Residual capacity
    int    rev;       ///< Index of the reverse edge in dst's adjacency list
};

/**
 * @brief Push-relabel max-flow graph.
 *
 * Build the graph with addTLink() and addNLink(), then call maxflow().
 * After maxflow(), query cut membership with inSourceSet().
 */
class MaxFlowGraph
{
public:
    /**
     * @brief Construct a graph for the given number of pixel nodes.
     * @param numPixels  Number of pixel nodes. Source = numPixels, Sink = numPixels+1.
     */
    explicit MaxFlowGraph(int numPixels);

    /**
     * @brief Reset the graph to empty (same node count), ready for a new solve.
     *        Cheaper than re-constructing for iterative GrabCut.
     */
    void reset(int numPixels);

    /**
     * @brief Add a terminal (T-link) capacity for pixel node i.
     * @param i          Pixel node index (0..numPixels-1)
     * @param toSource   Capacity of edge  source → i  (i.e. Pr(background) cost)
     * @param toSink     Capacity of edge  i → sink    (i.e. Pr(foreground) cost)
     *
     * Note: GrabCut convention — toSource = -log P(bg | x),
     *                             toSink   = -log P(fg | x).
     * Forced foreground: toSource = INF, toSink = 0.
     * Forced background: toSource = 0,   toSink = INF.
     */
    void addTLink(int i, double toSource, double toSink);

    /**
     * @brief Add an undirected N-link (smoothness) edge between pixels i and j.
     * @param i     First pixel node
     * @param j     Second pixel node
     * @param cap   Capacity in both directions (symmetric)
     */
    void addNLink(int i, int j, double cap);

    /**
     * @brief Run the max-flow algorithm.
     * @return The value of the maximum flow (= minimum cut energy).
     */
    double maxflow();

    /**
     * @brief Mark which nodes are reachable from source in the residual graph.
     *        Must be called once after maxflow() before using inSourceSet().
     *        GrabCut calls this automatically.
     */
    void bfsMarkSource();

    /**
     * @brief Query cut membership after maxflow() + bfsMarkSource().
     * @param i  Pixel node index
     * @return   true  → node is on the SOURCE side (foreground)
     *           false → node is on the SINK side   (background)
     */
    bool inSourceSet(int i) const;

    static constexpr double INF = 1e18;

private:
    int numNodes; ///< Total nodes = numPixels + 2 (source + sink)
    int source;   ///< Source node index
    int sink;     ///< Sink node index

    // Adjacency list: adj[v] is the list of edges leaving v
    std::vector<std::vector<Edge>> adj;

    // Push-relabel state
    std::vector<double> excess;   ///< Excess flow at each node
    std::vector<int>    height;   ///< Height label
    std::vector<int>    current;  ///< Current arc pointer (for gap heuristic)
    std::vector<int>    heightCount; ///< Count of nodes at each height (gap heuristic)

    // BFS reachability from source (for cut query)
    std::vector<bool>   visited;

    void push(int u, Edge &e);
    void relabel(int u);
    void discharge(int u);
    void bfsMarkSource();
    void globalRelabel();

    // FIFO active node queue for highest-label selection
    std::vector<std::vector<int>> buckets;
    int highestActive;

    int addEdge(int u, int v, double cap);
};

#endif // MAXFLOW_H
