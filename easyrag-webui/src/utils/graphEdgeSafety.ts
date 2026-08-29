import type { UndirectedGraph } from 'graphology'

/**
 * Prototype-safe replacement for `graph.hasEdge(source, target)`.
 * Works around graphology bugs where node ids can collide with JS internals.
 */
export const hasEdgeSafe = (graph: UndirectedGraph, source: string, target: string): boolean => {
  if (!graph.hasNode(source) || !graph.hasNode(target)) return false
  const edgeKey = graph.edge(source, target)
  return edgeKey !== undefined && graph.hasEdge(edgeKey)
}

/**
 * Add an undirected edge safely, working around graphology prototype collision bugs.
 * Returns the dynamic edge id, or null if both orientations fail.
 */
export const addUndirectedEdgeSafe = (
  graph: UndirectedGraph,
  source: string,
  target: string,
  attributes: Record<string, unknown>
): string | null => {
  try {
    return graph.addEdge(source, target, attributes)
  } catch {
    try {
      return graph.addEdge(target, source, attributes)
    } catch {
      return null
    }
  }
}
