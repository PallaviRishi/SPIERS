/**
 * @file
 * Header: Max-flow / Min-cut Graph Engine
 *
 * Implements highest-label push-relabel max-flow for graph cut segmentation.
 * Based on the algorithm described in:
 *   Boykov & Kolmogorov, "An Experimental Comparison of Min-Cut/Max-Flow
 *   Algorithms for Energy Minimization in Vision", PAMI 2004.
 *
 * The graph has:
 *   - One node per image pixel  (indices 0 .. N-1)
 *   - One source node           (index N)
 *   - One sink node             (index N+1)
 *
 * T-links : pixel <-> source and pixel <-> sink  (unary / data terms)
 * N-links : pixel <-> pixel                      (pairwise / smoothness terms)
 *
 * All edge capacities are non-negative doubles.
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2026 by the SPIERS contributors.
 */

#ifndef MAXFLOW_H
#define MAXFLOW_H

#include <cstdint>
#include <limits>
#include <vector>

/**
 * @brief One directed edge in the residual graph.
 */
struct Edge
{
    double cap;  ///< Residual capacity
    int dst;     ///< Destination node index
    int rev;     ///< Index of the reverse edge in dst's adjacency list
};

/**
 * @brief Push-relabel max-flow graph.
 *
 * Build the graph with addTLink() and addNLink(), then call maxflow().
 * After maxflow(), call bfsMarkSource() once, then query each pixel node
 * with inSourceSet().
 */
class MaxFlowGraph
{
public:
    /**
     * @brief Construct a graph for the given number of pixel nodes.
     * @param numPixels  Number of pixel nodes.
     *                   Source node = numPixels, sink node = numPixels + 1.
     */
    explicit MaxFlowGraph(int numPixels);

    /**
     * @brief Reset the graph to empty with the same node count, ready for a
     *        new solve. Cheaper than re-constructing for iterative GrabCut.
     * @param numPixels  Number of pixel nodes.
     */
    void reset(int numPixels);

    /**
     * @brief Add terminal (T-link) capacities for pixel node i.
     *
     * GrabCut convention:
     *   toSource = -log P(bg | z)   (cost of assigning to background)
     *   toSink   = -log P(fg | z)   (cost of assigning to foreground)
     *   Forced foreground: toSource = INF, toSink = 0.
     *   Forced background: toSource = 0,   toSink = INF.
     *
     * @param i         Pixel node index (0 .. numPixels-1).
     * @param toSource  Capacity of edge source -> i.
     * @param toSink    Capacity of edge i -> sink.
     */
    void addTLink(int i, double toSource, double toSink);

    /**
     * @brief Add an undirected N-link (smoothness) edge between pixels i and j.
     * @param i    First pixel node index.
     * @param j    Second pixel node index.
     * @param cap  Capacity in both directions (symmetric).
     */
    void addNLink(int i, int j, double cap);

    /**
     * @brief Run the max-flow algorithm.
     * @return The value of the maximum flow (equals the minimum cut energy).
     */
    double maxflow();

    /**
     * @brief Mark which nodes are reachable from source in the residual graph.
     *        Must be called once after maxflow() and before inSourceSet().
     */
    void bfsMarkSource();

    /**
     * @brief Query cut membership after maxflow() and bfsMarkSource().
     * @param i  Pixel node index.
     * @return   true  if the node is on the source side (foreground).
     *           false if the node is on the sink side   (background).
     */
    bool inSourceSet(int i) const;

    static constexpr double INF = 1e18; ///< Capacity used for forced terminals

private:
    int numNodes; ///< Total nodes = numPixels + 2 (source + sink)
    int source;   ///< Source node index
    int sink;     ///< Sink node index

    std::vector<std::vector<Edge>> adj; ///< Adjacency list: adj[v] lists edges leaving v

    std::vector<double> excess;      ///< Excess flow at each node
    std::vector<int> current;        ///< Current arc pointer (advance-arc heuristic)
    std::vector<int> height;         ///< Height label per node
    std::vector<int> heightCount;    ///< Count of nodes at each height (gap heuristic)
    std::vector<bool> visited;       ///< BFS reachability from source (for cut query)

    std::vector<std::vector<int>> buckets; ///< Per-height active node lists
    int highestActive;                     ///< Highest non-empty bucket index

    /**
     * @brief Add a single directed edge u -> v with the given capacity.
     *        Also adds the residual reverse edge v -> u with capacity 0.
     * @return Index of the reverse edge in adj[v] (used to set up residual).
     */
    int addEdge(int u, int v, double cap);

    /**
     * @brief Push as much excess as possible along edge e from node u.
     * @param u  Source node of the push.
     * @param e  Edge to push along (modified in place).
     */
    void push(int u, Edge &e);

    /**
     * @brief Relabel node u: raise its height to one above the lowest
     *        neighbour reachable via a residual edge.
     * @param u  Node to relabel.
     */
    void relabel(int u);

    /**
     * @brief Saturate all admissible edges from u until its excess is zero.
     *        Relabels u if needed. Applies the gap heuristic.
     * @param u  Active node to discharge.
     */
    void discharge(int u);

    /**
     * @brief Perform a reverse BFS from the sink to compute exact height labels.
     *        Called once at initialisation and periodically during the main loop.
     */
    void globalRelabel();
};

#endif // MAXFLOW_H
