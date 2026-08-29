export const controlButtonVariant = 'ghost' as const

export const labelColorDarkTheme = '#FFFFFF'
export const LabelColorHighlightedDarkTheme = '#000000'
export const labelColorLightTheme = '#000'

export const nodeColorDisabled = '#E2E2E2'
export const nodeBorderColor = '#EEEEEE'
export const nodeBorderColorSelected = '#F57F17'

export const edgeColorDarkTheme = '#888888'
export const edgeColorSelected = '#F57F17'
export const edgeColorHighlightedDarkTheme = '#F57F17'
export const edgeColorHighlightedLightTheme = '#F57F17'

export const searchResultLimit = 50
export const labelListLimit = 100

export const searchHistoryMaxItems = 500
export const searchHistoryVersion = '1.0'

export const popularLabelsDefaultLimit = 300
export const searchLabelsDefaultLimit = 50

export const dropdownDisplayLimit = 300

export const minNodeSize = 4
export const maxNodeSize = 20

export const healthCheckInterval = 15 // seconds

export const defaultQueryLabel = '*'

export const supportedFileTypes = {
  'text/plain': [
    '.txt', '.md', '.mdx', '.rtf', '.odt', '.tex', '.epub',
    '.html', '.htm', '.csv', '.json', '.xml', '.yaml', '.yml',
    '.log', '.conf', '.ini', '.properties', '.sql', '.bat', '.sh',
    '.c', '.h', '.cpp', '.hpp', '.py', '.java', '.js', '.ts',
    '.swift', '.go', '.rb', '.php', '.css', '.scss', '.less'
  ],
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
}

export const SiteInfo = {
  name: 'EasyRAG',
  home: '/',
  github: 'https://github.com/easyrag/easy-rag'
}

export const LABEL_RENDER_LIMIT = 2000
export const ANIMATE_NODE_LIMIT = 5000
export const EDGE_PERF_LIMIT = 5000

export const workerBudgetMs = (order: number): number => Math.min(1500 + order / 10, 10000)
