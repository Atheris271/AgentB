#!/usr/bin/env node
/**
 * Phase 1 -- Graph Topology Analysis Script
 * Analyzes a codebase knowledge graph to surface structural signals
 * useful for designing a pedagogical tour.
 */

const fs = require('fs');
const path = require('path');

// ── CLI args ──────────────────────────────────────────────────────
const inputPath  = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  console.error('Usage: node ua-tour-analyze.js <input.json> <output.json>');
  process.exit(1);
}

let data;
try {
  data = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
} catch (e) {
  console.error('Failed to read or parse input file:', e.message);
  process.exit(1);
}

const { nodes, edges, layers } = data;

// ── Helpers ───────────────────────────────────────────────────────
const nodeSet = new Set(nodes.map(n => n.id));
const nodeMap = new Map(nodes.map(n => [n.id, n]));

// Filter edges to only file-level <-> file-level for structural analysis
const fileEdges = edges.filter(e => nodeSet.has(e.source) && nodeSet.has(e.target));

// ── A. Fan-In Ranking ─────────────────────────────────────────────
const fanIn = new Map();
nodes.forEach(n => fanIn.set(n.id, 0));
fileEdges.forEach(e => {
  fanIn.set(e.target, (fanIn.get(e.target) || 0) + 1);
});
const fanInRanking = Array.from(fanIn.entries())
  .map(([id, count]) => ({ id, fanIn: count, name: nodeMap.get(id)?.name || id }))
  .sort((a, b) => b.fanIn - a.fanIn)
  .slice(0, 20);

// ── B. Fan-Out Ranking ────────────────────────────────────────────
const fanOut = new Map();
nodes.forEach(n => fanOut.set(n.id, 0));
fileEdges.forEach(e => {
  fanOut.set(e.source, (fanOut.get(e.source) || 0) + 1);
});
const fanOutRanking = Array.from(fanOut.entries())
  .map(([id, count]) => ({ id, fanOut: count, name: nodeMap.get(id)?.name || id }))
  .sort((a, b) => b.fanOut - a.fanOut)
  .slice(0, 20);

// ── C. Entry Point Candidates ─────────────────────────────────────
const fanOutValues = Array.from(fanOut.values());
const fanOutThreshold = fanOutValues.length > 0
  ? fanOutValues.sort((a, b) => a - b)[Math.floor(fanOutValues.length * 0.9)]
  : 0;
const fanInValues = Array.from(fanIn.values());
const fanInBottomThreshold = fanInValues.length > 0
  ? fanInValues.sort((a, b) => a - b)[Math.floor(fanInValues.length * 0.25)]
  : 0;

const codeEntryNames = new Set([
  'index.ts', 'index.js', 'main.ts', 'main.js', 'app.ts', 'app.js',
  'server.ts', 'server.js', 'mod.rs', 'main.go', 'main.py', 'main.rs',
  'manage.py', 'app.py', 'wsgi.py', 'asgi.py', 'run.py', '__main__.py',
  'Application.java', 'Main.java', 'Program.cs', 'config.ru', 'index.php',
  'App.swift', 'Application.kt', 'main.cpp', 'main.c'
]);

function entryScore(node) {
  let score = 0;
  const fp = node.filePath || '';
  const name = node.name || '';

  // Code file scoring
  if (node.type === 'file') {
    if (codeEntryNames.has(name)) score += 3;
    const depth = fp.split(path.sep).filter(s => s.length > 0).length;
    if (depth <= 1 || (fp.startsWith('code/') && depth <= 2)) score += 1;
    const fo = fanOut.get(node.id) || 0;
    if (fo >= fanOutThreshold && fanOutValues.length > 0) score += 1;
    const fi = fanIn.get(node.id) || 0;
    if (fi <= fanInBottomThreshold && fanInValues.length > 0) score += 1;
  }

  // Documentation scoring
  if (node.type === 'document') {
    if (name === 'README.md' && (fp === 'README.md' || fp.endsWith('/README.md') || fp === '')) score += 5;
    else if (name.endsWith('.md') && (depthSlashes(fp) <= 1)) score += 2;
  }

  return score;
}

function depthSlashes(fp) {
  return fp.split('/').filter(s => s.length > 0).length;
}

const allScores = nodes.map(n => ({
  id: n.id,
  score: entryScore(n),
  name: n.name,
  summary: n.summary || ''
}));
allScores.sort((a, b) => b.score - a.score);
const entryPointCandidates = allScores.filter(e => e.score > 0).slice(0, 5);

// ── D. BFS Traversal from Top Code Entry Point ────────────────────
// Skip documentation entries, find first code entry point
const topCodeEntry = entryPointCandidates.find(e => {
  const n = nodeMap.get(e.id);
  return n && n.type === 'file';
}) || entryPointCandidates[0];

const bfsStartNode = topCodeEntry ? topCodeEntry.id : nodes[0]?.id;

// Build adjacency list for BFS using imports & depends_on edges (file-level)
const adjacency = new Map();
nodes.forEach(n => adjacency.set(n.id, []));
fileEdges.forEach(e => {
  if (e.type === 'imports' || e.type === 'depends_on') {
    const adj = adjacency.get(e.source);
    if (adj && !adj.includes(e.target)) {
      adj.push(e.target);
    }
  }
});

function bfs(start) {
  const visited = new Set();
  const queue = [{ id: start, depth: 0 }];
  const order = [];
  const depthMap = {};
  const byDepth = {};

  visited.add(start);

  while (queue.length > 0) {
    const { id, depth } = queue.shift();
    order.push(id);
    depthMap[id] = depth;
    if (!byDepth[depth]) byDepth[depth] = [];
    byDepth[depth].push(id);

    const neighbors = adjacency.get(id) || [];
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor) && nodeSet.has(neighbor)) {
        visited.add(neighbor);
        queue.push({ id: neighbor, depth: depth + 1 });
      }
    }
  }

  return { order, depthMap, byDepth };
}

const bfsTraversal = bfsStartNode ? bfs(bfsStartNode) : { order: [], depthMap: {}, byDepth: {} };

// ── E. Non-Code File Inventory ────────────────────────────────────
const nonCodeFiles = {
  documentation: [],
  infrastructure: [],
  data: [],
  config: []
};

const typeToCategory = {
  'document': 'documentation',
  'service': 'infrastructure',
  'pipeline': 'infrastructure',
  'resource': 'infrastructure',
  'table': 'data',
  'schema': 'data',
  'endpoint': 'data',
  'config': 'config'
};

nodes.forEach(n => {
  const cat = typeToCategory[n.type];
  if (cat) {
    nonCodeFiles[cat].push({
      id: n.id,
      name: n.name,
      summary: n.summary || ''
    });
  }
});

// ── F. Tightly Coupled Clusters ───────────────────────────────────
// Find bidirectional pairs and expand into clusters
// Only consider file nodes with imports/depends_on edges
const clusterEdgeTypes = ['imports', 'depends_on'];
const clusterEdges = fileEdges.filter(e => clusterEdgeTypes.includes(e.type));

// Build reverse adjacency
const revAdj = new Map();
nodes.forEach(n => revAdj.set(n.id, new Set()));
clusterEdges.forEach(e => {
  if (revAdj.has(e.target)) revAdj.get(e.target).add(e.source);
});

// Find pairs with bidirectional edges
const bidirPairs = [];
const seen = new Set();
clusterEdges.forEach(e => {
  const key = [e.source, e.target].sort().join('|||');
  if (seen.has(key)) return;
  if (revAdj.has(e.source) && revAdj.get(e.source).has(e.target)) {
    bidirPairs.push([e.source, e.target]);
    seen.add(key);
  }
});

// Expand into clusters
const clusters = [];
const clustered = new Set();

bidirPairs.forEach(pair => {
  const clusterSet = new Set(pair);
  // Expand by adding nodes connected to 2+ existing members
  let changed = true;
  while (changed) {
    changed = false;
    nodes.forEach(n => {
      if (clusterSet.has(n.id)) return;
      let connections = 0;
      const neighbors = adjacency.get(n.id) || [];
      for (const nb of neighbors) {
        if (clusterSet.has(nb)) connections++;
        if (connections >= 2) break;
      }
      // Also check reverse connections
      for (const [src, tgts] of adjacency.entries()) {
        if (tgts.includes(n.id) && clusterSet.has(src)) connections++;
        if (connections >= 2) break;
      }
      if (connections >= 2) {
        clusterSet.add(n.id);
        changed = true;
      }
    });
  }
  if (clusterSet.size >= 2 && clusterSet.size <= 5) {
    const clusterNodes = Array.from(clusterSet);
    // Count edges among cluster members
    let edgeCount = 0;
    clusterEdges.forEach(e => {
      if (clusterSet.has(e.source) && clusterSet.has(e.target)) edgeCount++;
    });
    clusters.push({ nodes: clusterNodes, edgeCount });
    clusterNodes.forEach(id => clustered.add(id));
  }
});

clusters.sort((a, b) => b.edgeCount - a.edgeCount);
const topClusters = clusters.slice(0, 10);

// ── G. Layer List ─────────────────────────────────────────────────
const layerData = {
  count: layers.length,
  list: layers.map(l => ({ id: l.id, name: l.name, description: l.description }))
};

// ── H. Node Summary Index ─────────────────────────────────────────
const nodeSummaryIndex = {};
nodes.forEach(n => {
  nodeSummaryIndex[n.id] = {
    name: n.name,
    type: n.type,
    summary: n.summary || ''
  };
});

// ── Output ────────────────────────────────────────────────────────
const result = {
  scriptCompleted: true,
  entryPointCandidates,
  fanInRanking,
  fanOutRanking,
  bfsTraversal: {
    startNode: bfsStartNode,
    order: bfsTraversal.order,
    depthMap: bfsTraversal.depthMap,
    byDepth: bfsTraversal.byDepth
  },
  nonCodeFiles,
  clusters: topClusters,
  layers: layerData,
  nodeSummaryIndex,
  totalNodes: nodes.length,
  totalEdges: edges.length
};

try {
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2), 'utf8');
  console.log('Analysis complete. Written to ' + outputPath);
  console.log('  Entry point candidates: ' + entryPointCandidates.length);
  console.log('  Fan-in ranking top:     ' + fanInRanking[0]?.name);
  console.log('  Fan-out ranking top:    ' + fanOutRanking[0]?.name);
  console.log('  BFS nodes reached:      ' + bfsTraversal.order.length);
  console.log('  Clusters found:         ' + topClusters.length);
  console.log('  Node summary entries:   ' + Object.keys(nodeSummaryIndex).length);
} catch (e) {
  console.error('Failed to write output:', e.message);
  process.exit(1);
}
