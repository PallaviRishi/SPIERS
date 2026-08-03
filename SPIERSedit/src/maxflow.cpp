/**
 * @file
 * Source: Max-flow / Min-cut Graph Engine
 *
 * Implements the highest-label push-relabel algorithm with gap heuristic and
 * periodic global relabelling via reverse BFS from the sink.
 *
 * Complexity: O(V^2 * sqrt(E)) in theory; in practice very fast on grid graphs
 * typical of image segmentation (hundreds of thousands of pixels).
 *
 * All SPIERSedit code is released under the GNU General Public License.
 * See LICENSE.md files in the programme directory.
 *
 * Copyright 2024 by the SPIERS contributors.
 */

#include "maxflow.h"
#include <algorithm>
#include <queue>
#include <cassert>
#include <cstring>

// ─── Construction ─────────────────────────────────────────────────────────────

MaxFlowGraph::MaxFlowGraph(int numPixels)
{
    reset(numPixels);
}

void MaxFlowGraph::reset(int numPixels)
{
    numNodes = numPixels + 2;
    source   = numPixels;
    sink     = numPixels + 1;

    adj.assign(static_cast<size_t>(numNodes), {});
    excess.assign(static_cast<size_t>(numNodes), 0.0);
    height.assign(static_cast<size_t>(numNodes), 0);
    current.assign(static_cast<size_t>(numNodes), 0);
    visited.assign(static_cast<size_t>(numNodes), false);
    heightCount.assign(static_cast<size_t>(numNodes * 2 + 2), 0);

    buckets.assign(static_cast<size_t>(numNodes * 2), {});
    highestActive = -1;
}

// ─── Edge helpers ─────────────────────────────────────────────────────────────

int MaxFlowGraph::addEdge(int u, int v, double cap)
{
    int idxUV = static_cast<int>(adj[static_cast<size_t>(u)].size());
    int idxVU = static_cast<int>(adj[static_cast<size_t>(v)].size());

    adj[static_cast<size_t>(u)].push_back({ v, cap, idxVU });
    adj[static_cast<size_t>(v)].push_back({ u, 0.0, idxUV });  // reverse edge (residual)

    return idxUV; // return index of reverse in v's list (unused but can be helpful)
}

void MaxFlowGraph::addTLink(int i, double toSource, double toSink)
{
    // source → i  with capacity toSource
    // i      → sink with capacity toSink
    addEdge(source, i, toSource);
    addEdge(i, sink,  toSink);
}

void MaxFlowGraph::addNLink(int i, int j, double cap)
{
    // Undirected: add forward and reverse both with capacity cap
    int idxIJ = static_cast<int>(adj[static_cast<size_t>(i)].size());
    int idxJI = static_cast<int>(adj[static_cast<size_t>(j)].size());

    adj[static_cast<size_t>(i)].push_back({ j, cap, idxJI });
    adj[static_cast<size_t>(j)].push_back({ i, cap, idxIJ });
}

// ─── Push-relabel core ────────────────────────────────────────────────────────

void MaxFlowGraph::push(int u, Edge &e)
{
    double delta = std::min(excess[static_cast<size_t>(u)], e.cap);
    if (delta <= 0.0) return;

    e.cap -= delta;
    adj[static_cast<size_t>(e.dst)][static_cast<size_t>(e.rev)].cap += delta;
    excess[static_cast<size_t>(u)]     -= delta;
    excess[static_cast<size_t>(e.dst)] += delta;
}

void MaxFlowGraph::relabel(int u)
{
    int minH = numNodes * 2;
    for (const Edge &e : adj[static_cast<size_t>(u)])
        if (e.cap > 0.0)
            minH = std::min(minH, height[static_cast<size_t>(e.dst)]);

    if (minH < numNodes * 2)
    {
        heightCount[static_cast<size_t>(height[static_cast<size_t>(u)])]--;
        height[static_cast<size_t>(u)] = minH + 1;
        heightCount[static_cast<size_t>(height[static_cast<size_t>(u)])]++;
    }
}

void MaxFlowGraph::discharge(int u)
{
    while (excess[static_cast<size_t>(u)] > 0.0)
    {
        if (current[static_cast<size_t>(u)] == static_cast<int>(adj[static_cast<size_t>(u)].size()))
        {
            // All edges exhausted for current label — relabel
            int oldH = height[static_cast<size_t>(u)];
            relabel(u);

            // Gap heuristic: if a gap appears at height oldH, all nodes above
            // it can be relabelled to numNodes (their cuts are on the sink side)
            if (heightCount[static_cast<size_t>(oldH)] == 0)
            {
                for (int v = 0; v < numNodes; v++)
                {
                    if (v == source || v == sink) continue;
                    if (height[static_cast<size_t>(v)] > oldH
                        && height[static_cast<size_t>(v)] < numNodes)
                    {
                        heightCount[static_cast<size_t>(height[static_cast<size_t>(v)])]--;
                        height[static_cast<size_t>(v)] = numNodes;
                        // Don't add to heightCount[numNodes] — these are dead
                    }
                }
            }
            current[static_cast<size_t>(u)] = 0;
        }
        else
        {
            Edge &e = adj[static_cast<size_t>(u)][static_cast<size_t>(current[static_cast<size_t>(u)])];
            if (e.cap > 0.0 && height[static_cast<size_t>(u)] == height[static_cast<size_t>(e.dst)] + 1)
                push(u, e);
            else
                current[static_cast<size_t>(u)]++;
        }
    }
}

// ─── Global relabel via reverse BFS from sink ─────────────────────────────────

void MaxFlowGraph::globalRelabel()
{
    // BFS on the REVERSE residual graph from sink
    // height[v] = shortest path length from v to sink in the residual graph
    std::fill(height.begin(), height.end(), numNodes); // numNodes = "infinity"
    std::fill(heightCount.begin(), heightCount.end(), 0);

    height[static_cast<size_t>(sink)] = 0;
    heightCount[0] = 1;

    std::queue<int> q;
    q.push(sink);

    while (!q.empty())
    {
        int v = q.front(); q.pop();
        for (int idx = 0; idx < static_cast<int>(adj[static_cast<size_t>(v)].size()); idx++)
        {
            const Edge &e = adj[static_cast<size_t>(v)][static_cast<size_t>(idx)];
            int u = e.dst;
            // reverse edge from v→u in original means edge u→v in reverse graph
            // The reverse edge in adj[v] points to u; it has residual cap if
            // adj[u][e.rev].cap > 0
            if (height[static_cast<size_t>(u)] == numNodes
                && adj[static_cast<size_t>(u)][static_cast<size_t>(e.rev)].cap > 0.0)
            {
                height[static_cast<size_t>(u)] = height[static_cast<size_t>(v)] + 1;
                heightCount[static_cast<size_t>(height[static_cast<size_t>(u)])]++;
                q.push(u);
            }
        }
    }

    height[static_cast<size_t>(source)] = numNodes; // source is above everything
}

// ─── Main max-flow entry point ────────────────────────────────────────────────

double MaxFlowGraph::maxflow()
{
    // ── Initialise: saturate all edges out of source ──────────────────────────
    height[static_cast<size_t>(source)] = numNodes;
    heightCount[static_cast<size_t>(numNodes)]++;

    excess[static_cast<size_t>(source)] = INF; // effectively infinite

    for (Edge &e : adj[static_cast<size_t>(source)])
    {
        push(source, e);
    }

    // Global relabel to get good initial heights
    globalRelabel();

    // Reset active node buckets
    for (auto &b : buckets) b.clear();
    highestActive = -1;

    for (int u = 0; u < numNodes; u++)
    {
        if (u == source || u == sink) continue;
        if (excess[static_cast<size_t>(u)] > 0.0)
        {
            int h = height[static_cast<size_t>(u)];
            if (h < numNodes)
            {
                buckets[static_cast<size_t>(h)].push_back(u);
                highestActive = std::max(highestActive, h);
            }
        }
    }

    int relabelCount  = 0;
    int globalRelabelFreq = std::max(1, numNodes / 2);

    // ── Main discharge loop ───────────────────────────────────────────────────
    while (highestActive >= 0)
    {
        // Pick the highest-label active node
        while (highestActive >= 0 && buckets[static_cast<size_t>(highestActive)].empty())
            highestActive--;

        if (highestActive < 0) break;

        int u = buckets[static_cast<size_t>(highestActive)].back();
        buckets[static_cast<size_t>(highestActive)].pop_back();

        if (excess[static_cast<size_t>(u)] <= 0.0) continue;
        if (u == source || u == sink) continue;

        int oldH = height[static_cast<size_t>(u)];
        discharge(u);
        relabelCount++;

        if (relabelCount % globalRelabelFreq == 0)
            globalRelabel();

        int newH = height[static_cast<size_t>(u)];
        if (excess[static_cast<size_t>(u)] > 0.0 && newH < numNodes)
        {
            buckets[static_cast<size_t>(newH)].push_back(u);
            highestActive = std::max(highestActive, newH);
        }
        (void)oldH;
    }

    // Total flow = excess at sink
    return excess[static_cast<size_t>(sink)];
}

// ─── Cut query ────────────────────────────────────────────────────────────────

/**
 * @brief BFS from source on residual graph to determine the source-side cut set.
 *        Call once after maxflow(). inSourceSet() is then valid.
 */
void MaxFlowGraph::bfsMarkSource()
{
    std::fill(visited.begin(), visited.end(), false);
    std::queue<int> q;
    q.push(source);
    visited[static_cast<size_t>(source)] = true;

    while (!q.empty())
    {
        int u = q.front(); q.pop();
        for (const Edge &e : adj[static_cast<size_t>(u)])
        {
            if (!visited[static_cast<size_t>(e.dst)] && e.cap > 0.0)
            {
                visited[static_cast<size_t>(e.dst)] = true;
                q.push(e.dst);
            }
        }
    }
}

bool MaxFlowGraph::inSourceSet(int i) const
{
    return visited[static_cast<size_t>(i)];
}
