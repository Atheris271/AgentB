#!/usr/bin/env node
/**
 * Phase 1 -- Structural Analysis Script
 * Reads the combined input JSON, computes structural patterns, writes results JSON.
 */

const fs = require('fs');
const path = require('path');

// --- Parse args ---
const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  console.error('Usage: node ua-arch-analyze.js <input.json> <output.json>');
  process.exit(1);
}

// --- Read input ---
let data;
try {
  data = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
} catch (e) {
  console.error('Failed to read/parse input file:', e.message);
  process.exit(1);
}

const { fileNodes, importEdges, allEdges } = data;
if (!fileNodes || !importEdges || !allEdges) {
  console.error('Input must contain fileNodes, importEdges, allEdges');
  process.exit(1);
}

// Helper: normalize path to forward slashes
function norm(p) { return p.replace(/\\/g, '/'); }

// All file-level node IDs
const allNodeIds = new Set(fileNodes.map(n => n.id));
const nodeMap = {};
fileNodes.forEach(n => { nodeMap[n.id] = n; });

// --- A. Directory Grouping ---
// Find common prefix
const filePaths = fileNodes.map(n => norm(n.filePath));
let commonPrefix = '';
if (filePaths.length > 0) {
  const first = filePaths[0];
  for (let i = 0; i < first.length; i++) {
    const ch = first[i];
    if (filePaths.every(p => p[i] === ch)) {
      commonPrefix += ch;
    } else {
      break;
    }
  }
  // Truncate to last '/' to get directory-level prefix
  const lastSlash = commonPrefix.lastIndexOf('/');
  if (lastSlash >= 0) {
    commonPrefix = commonPrefix.substring(0, lastSlash + 1);
  } else {
    commonPrefix = '';
  }
}

// Group by first directory segment after common prefix
const directoryGroups = {};
fileNodes.forEach(n => {
  const fp = norm(n.filePath);
  let relPath = commonPrefix ? fp.substring(commonPrefix.length) : fp;
  if (relPath.startsWith('/')) relPath = relPath.substring(1);
  const parts = relPath.split('/');
  let groupName;
  if (parts.length >= 2 || (parts.length === 1 && relPath.includes('/'))) {
    groupName = parts[0];
  } else {
    // Flat structure: group by extension pattern
    const ext = path.extname(fp).toLowerCase();
    const base = path.basename(fp);
    if (base.match(/\.test\./) || base.match(/\.spec\./) || base.match(/^test_/) || base.match(/_test\./)) {
      groupName = 'test';
    } else if (base.match(/\.config\./) || base.match(/^\./) || base === 'package.json') {
      groupName = 'config';
    } else {
      groupName = 'root';
    }
  }

  // Further refine for nested groups
  if (groupName === 'data' && parts.length >= 2) {
    const sub = parts[1];
    // Group some data subdirectories together
    if (['b1_fixtures', 'memory_inputs', 'messages', 'tool_inputs', 'docs', 'tables'].includes(sub)) {
      groupName = 'data';
    }
  }

  if (!directoryGroups[groupName]) directoryGroups[groupName] = [];
  directoryGroups[groupName].push(n.id);
});

// --- B. Node Type Grouping ---
const nodeTypeGroups = {};
fileNodes.forEach(n => {
  if (!nodeTypeGroups[n.type]) nodeTypeGroups[n.type] = [];
  nodeTypeGroups[n.type].push(n.id);
});

// --- C. Import Adjacency Matrix ---
const fanOut = {}; // which files a file imports
const fanIn = {};  // which files import a file
fileNodes.forEach(n => {
  fanOut[n.id] = [];
  fanIn[n.id] = [];
});

importEdges.forEach(e => {
  if (allNodeIds.has(e.source) && allNodeIds.has(e.target)) {
    if (!fanOut[e.source].includes(e.target)) fanOut[e.source].push(e.target);
    if (!fanIn[e.target].includes(e.source)) fanIn[e.target].push(e.source);
  }
});

const fileFanIn = {};
const fileFanOut = {};
Object.keys(fanIn).forEach(id => { fileFanIn[id] = fanIn[id].length; });
Object.keys(fanOut).forEach(id => { fileFanOut[id] = fanOut[id].length; });

// Directory-group-level adjacency
const groupImports = {};
const groupImportedBy = {};
Object.keys(directoryGroups).forEach(g => {
  groupImports[g] = new Set();
  groupImportedBy[g] = new Set();
});

importEdges.forEach(e => {
  if (!allNodeIds.has(e.source) || !allNodeIds.has(e.target)) return;
  const srcGroup = Object.keys(directoryGroups).find(g => directoryGroups[g].includes(e.source));
  const tgtGroup = Object.keys(directoryGroups).find(g => directoryGroups[g].includes(e.target));
  if (srcGroup && tgtGroup && srcGroup !== tgtGroup) {
    groupImports[srcGroup].add(tgtGroup);
    groupImportedBy[tgtGroup].add(srcGroup);
  }
});

// --- D. Cross-Category Dependency Analysis ---
// Filter allEdges to only file-level edges (between file-level nodes)
const fileLevelEdges = allEdges.filter(e => {
  return allNodeIds.has(e.source) && allNodeIds.has(e.target);
});

const crossCategoryEdges = [];
const catCounts = {};
fileLevelEdges.forEach(e => {
  const srcNode = nodeMap[e.source];
  const tgtNode = nodeMap[e.target];
  if (!srcNode || !tgtNode) return;
  const key = `${srcNode.type}|${e.type}|${tgtNode.type}`;
  if (!catCounts[key]) catCounts[key] = { fromType: srcNode.type, toType: tgtNode.type, edgeType: e.type, count: 0 };
  catCounts[key].count++;
  // Also compute between non-code and code nodes
});
// Also count all cross-category from allEdges (including non-file nodes)
const allCrossCat = {};
allEdges.forEach(e => {
  const srcNode = nodeMap[e.source] || { type: e.source.split(':')[0] };
  const tgtNode = nodeMap[e.target] || { type: e.target.split(':')[0] };
  const key = `${srcNode.type}|${e.type}|${tgtNode.type}`;
  if (!allCrossCat[key]) allCrossCat[key] = { fromType: srcNode.type, toType: tgtNode.type, edgeType: e.type, count: 0 };
  allCrossCat[key].count++;
});

// --- E. Inter-Group Import Frequency ---
const interGroupImports = {};
importEdges.forEach(e => {
  if (!allNodeIds.has(e.source) || !allNodeIds.has(e.target)) return;
  const srcGroup = Object.keys(directoryGroups).find(g => directoryGroups[g].includes(e.source));
  const tgtGroup = Object.keys(directoryGroups).find(g => directoryGroups[g].includes(e.target));
  if (srcGroup && tgtGroup && srcGroup !== tgtGroup) {
    const key = `${srcGroup} -> ${tgtGroup}`;
    interGroupImports[key] = (interGroupImports[key] || 0) + 1;
  }
});
const interGroupImportsArr = Object.entries(interGroupImports).map(([k, v]) => {
  const [from, to] = k.split(' -> ');
  return { from, to, count: v };
});

// --- F. Intra-Group Import Density ---
const intraGroupDensity = {};
Object.entries(directoryGroups).forEach(([groupName, nodeIds]) => {
  const idSet = new Set(nodeIds);
  let internalEdges = 0;
  let totalEdges = 0;
  nodeIds.forEach(id => {
    (fanOut[id] || []).forEach(tgt => {
      if (idSet.has(tgt)) internalEdges++;
      totalEdges++;
    });
    (fanIn[id] || []).forEach(src => {
      if (!idSet.has(src)) totalEdges++;
    });
  });
  // Count unique total edges correctly
  let uniqueExternalOut = 0;
  let uniqueExternalIn = 0;
  nodeIds.forEach(id => {
    (fanOut[id] || []).forEach(tgt => {
      if (!idSet.has(tgt)) uniqueExternalOut++;
    });
  });
  nodeIds.forEach(id => {
    (fanIn[id] || []).forEach(src => {
      if (!idSet.has(src)) uniqueExternalIn++;
    });
  });
  totalEdges = internalEdges + uniqueExternalOut + uniqueExternalIn;
  intraGroupDensity[groupName] = {
    internalEdges,
    totalEdges: Math.max(totalEdges, 1),
    density: totalEdges > 0 ? internalEdges / totalEdges : 0
  };
});

// --- G. Directory Pattern Matching ---
const patternMap = {
  'routes': 'api', 'api': 'api', 'controllers': 'api', 'endpoints': 'api', 'handlers': 'api',
  'routers': 'api', 'controller': 'api', 'blueprints': 'api', 'serializers': 'api',
  'services': 'service', 'core': 'service', 'lib': 'service', 'domain': 'service', 'logic': 'service',
  'internal': 'service', 'composables': 'service', 'mailers': 'service', 'jobs': 'service', 'channels': 'service',
  'signals': 'service',
  'models': 'data', 'db': 'data', 'data': 'data', 'persistence': 'data', 'repository': 'data', 'entities': 'data',
  'entity': 'data', 'migrations': 'data', 'sql': 'data', 'database': 'data', 'schema': 'data',
  'components': 'ui', 'views': 'ui', 'pages': 'ui', 'ui': 'ui', 'layouts': 'ui', 'screens': 'ui',
  'middleware': 'middleware', 'plugins': 'middleware', 'interceptors': 'middleware', 'guards': 'middleware',
  'utils': 'utility', 'helpers': 'utility', 'common': 'utility', 'shared': 'utility', 'tools': 'utility',
  'templatetags': 'utility', 'pkg': 'utility',
  'config': 'config', 'constants': 'config', 'env': 'config', 'settings': 'config',
  'management': 'config', 'commands': 'config',
  'tests': 'test', 'test': 'test', 'spec': 'test', 'specs': 'test',
  'types': 'types', 'interfaces': 'types', 'schemas': 'types', 'contracts': 'types', 'dtos': 'types',
  'hooks': 'hooks', 'state': 'state', 'store': 'state', 'reducers': 'state', 'actions': 'state', 'slices': 'state',
  'assets': 'assets', 'static': 'assets', 'public': 'assets',
  'cmd': 'entry', 'bin': 'entry',
  'docs': 'documentation', 'documentation': 'documentation', 'wiki': 'documentation',
  'deploy': 'infrastructure', 'deployment': 'infrastructure', 'infra': 'infrastructure', 'infrastructure': 'infrastructure',
  'docker': 'infrastructure', 'k8s': 'infrastructure', 'kubernetes': 'infrastructure', 'helm': 'infrastructure',
  'terraform': 'infrastructure', 'tf': 'infrastructure',
};

// Also check file-level patterns
function getFilePattern(node) {
  const fp = norm(node.filePath);
  const base = path.basename(fp).toLowerCase();

  if (base.match(/\.test\./) || base.match(/\.spec\./) || base.match(/^test_/) || base.match(/_test\./))
    return 'test';
  if (base.match(/\.d\.ts$/)) return 'types';
  if (['index.ts','index.js','__init__.py'].includes(base)) return 'entry';
  if (base === 'manage.py') return 'entry';
  if (base === 'wsgi.py' || base === 'asgi.py') return 'config';
  if (base === 'main.go' && fp.includes('/cmd/')) return 'entry';
  if (base === 'main.rs' || base === 'lib.rs') return 'entry';
  if (base === 'application.java' || base === 'program.cs') return 'entry';
  if (base === 'config.ru') return 'entry';
  if (['cargo.toml','go.mod','gemfile','pom.xml','build.gradle','composer.json'].includes(base)) return 'config';
  if (base === 'dockerfile' || base.startsWith('docker-compose')) return 'infrastructure';
  if (base.endsWith('.tf') || base.endsWith('.tfvars')) return 'infrastructure';
  if (base.endsWith('.sql')) return 'data';
  if (base.endsWith('.graphql') || base.endsWith('.gql') || base.endsWith('.proto')) return 'types';
  if (base.endsWith('.md') || base.endsWith('.rst')) return 'documentation';
  if (base === 'makefile') return 'infrastructure';

  // Check CI/CD
  if (fp.includes('.github/workflows/') || fp.includes('.gitlab-ci.yml') || fp.includes('jenkinsfile'))
    return 'ci-cd';

  return null;
}

const patternMatches = {};
Object.keys(directoryGroups).forEach(g => {
  const lower = g.toLowerCase();
  patternMatches[g] = patternMap[lower] || getFilePattern(nodeMap[directoryGroups[g][0]]) || 'utility';
});

// --- H. Deployment Topology Detection ---
const deploymentTopology = {
  hasDockerfile: false,
  hasCompose: false,
  hasK8s: false,
  hasTerraform: false,
  hasCI: false,
  infraFiles: []
};

fileNodes.forEach(n => {
  const fp = norm(n.filePath).toLowerCase();
  const base = path.basename(fp);
  if (base === 'dockerfile') { deploymentTopology.hasDockerfile = true; deploymentTopology.infraFiles.push(n.filePath); }
  if (base.startsWith('docker-compose')) { deploymentTopology.hasCompose = true; deploymentTopology.infraFiles.push(n.filePath); }
  if (fp.includes('.github/workflows/') || base === '.gitlab-ci.yml' || base === 'jenkinsfile') {
    deploymentTopology.hasCI = true;
    deploymentTopology.infraFiles.push(n.filePath);
  }
  if (base.endsWith('.tf')) { deploymentTopology.hasTerraform = true; deploymentTopology.infraFiles.push(n.filePath); }
  if (fp.includes('/k8s/') || fp.includes('/kubernetes/') || fp.includes('/helm/')) {
    deploymentTopology.hasK8s = true;
  }
});

// --- I. Data Pipeline Detection ---
const dataPipeline = {
  schemaFiles: [],
  migrationFiles: [],
  dataModelFiles: [],
  apiHandlerFiles: []
};

fileNodes.forEach(n => {
  const fp = norm(n.filePath).toLowerCase();
  if (fp.endsWith('.sql')) dataPipeline.schemaFiles.push(n.filePath);
  if (fp.endsWith('.graphql') || fp.endsWith('.proto') || fp.endsWith('.gql')) dataPipeline.schemaFiles.push(n.filePath);
  if (fp.includes('/migrations/') && fp.endsWith('.sql')) dataPipeline.migrationFiles.push(n.filePath);
  if (fp.includes('/models/') || fp.includes('/entities/') || fp.includes('/data/')) dataPipeline.dataModelFiles.push(n.filePath);
  if (fp.includes('/routes/') || fp.includes('/api/') || fp.includes('/controllers/') || fp.includes('/endpoints/') || fp.includes('/handlers/'))
    dataPipeline.apiHandlerFiles.push(n.filePath);
});

// Also check for schema configs, tool schemas
fileNodes.forEach(n => {
  if (n.tags && n.tags.some(t => t.includes('schema') || t.includes('data-model') || t.includes('message-contract'))) {
    if (!dataPipeline.dataModelFiles.includes(n.filePath)) dataPipeline.dataModelFiles.push(n.filePath);
  }
  if (n.tags && n.tags.some(t => t.includes('api') || t.includes('handler'))) {
    if (!dataPipeline.apiHandlerFiles.includes(n.filePath)) dataPipeline.apiHandlerFiles.push(n.filePath);
  }
});

// --- J. Documentation Coverage ---
const docCoverage = {
  groupsWithDocs: 0,
  totalGroups: Object.keys(directoryGroups).length,
  coverageRatio: 0,
  undocumentedGroups: []
};

// Check for README.md in each directory and doc files that reference code groups
Object.entries(directoryGroups).forEach(([groupName, nodeIds]) => {
  let hasDoc = false;
  nodeIds.forEach(id => {
    const n = nodeMap[id];
    if (!n) return;
    const fp = norm(n.filePath).toLowerCase();
    if (fp.endsWith('readme.md')) hasDoc = true;
    if (n.type === 'document') hasDoc = true;
  });
  if (hasDoc) {
    docCoverage.groupsWithDocs++;
  } else {
    docCoverage.undocumentedGroups.push(groupName);
  }
});
docCoverage.coverageRatio = docCoverage.totalGroups > 0
  ? Math.round((docCoverage.groupsWithDocs / docCoverage.totalGroups) * 100) / 100
  : 0;

// --- K. Dependency Direction ---
const dependencyDirection = [];
Object.entries(groupImports).forEach(([group, imports]) => {
  imports.forEach(tgt => {
    // Check if tgt also imports group
    const tgtImports = groupImports[tgt] || new Set();
    const fromTo = interGroupImports[`${group} -> ${tgt}`] || 0;
    const toFrom = interGroupImports[`${tgt} -> ${group}`] || 0;
    if (fromTo > toFrom) {
      dependencyDirection.push({ dependent: group, dependsOn: tgt });
    }
    // If equal, not recorded as direction is ambiguous
  });
});

// --- fileStats ---
const filesPerGroup = {};
Object.entries(directoryGroups).forEach(([g, ids]) => { filesPerGroup[g] = ids.length; });

const nodeTypeCounts = {};
Object.entries(nodeTypeGroups).forEach(([t, ids]) => { nodeTypeCounts[t] = ids.length; });

const fileStats = {
  totalFileNodes: fileNodes.length,
  filesPerGroup,
  nodeTypeCounts
};

// --- Build output ---
const output = {
  scriptCompleted: true,
  directoryGroups,
  nodeTypeGroups,
  crossCategoryEdges: Object.values(allCrossCat),
  interGroupImports: interGroupImportsArr,
  intraGroupDensity,
  patternMatches,
  deploymentTopology,
  dataPipeline,
  docCoverage,
  dependencyDirection,
  fileStats,
  fileFanIn,
  fileFanOut
};

// --- Write output ---
try {
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2), 'utf8');
} catch (e) {
  console.error('Failed to write output:', e.message);
  process.exit(1);
}

console.log(`Analysis complete. ${fileNodes.length} file nodes analyzed across ${Object.keys(directoryGroups).length} directory groups.`);
process.exit(0);
