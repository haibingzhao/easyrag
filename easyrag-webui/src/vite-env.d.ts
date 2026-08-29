/// <reference types="vite/client" />

declare module '*.css' {
  const content: Record<string, string>
  export default content
}

declare module 'react-syntax-highlighter' {
  import { Component, ReactNode } from 'react'

  export interface SyntaxHighlighterProps {
    language?: string
    style?: Record<string, any>
    customStyle?: Record<string, any>
    codeTagProps?: Record<string, any>
    useInlineStyles?: boolean
    showLineNumbers?: boolean
    showInlineLineNumbers?: boolean
    startingLineNumber?: number
    lineNumberContainerStyle?: Record<string, any>
    lineNumberStyle?: Record<string, any> | ((lineNumber: number) => Record<string, any>)
    wrapLines?: boolean
    wrapLongLines?: boolean
    lineProps?: Record<string, any> | ((lineNumber: number) => Record<string, any>)
    renderer?: (props: Record<string, any>) => ReactNode
    PreTag?: any
    CodeTag?: any
    children: string | string[]
    [key: string]: any
  }

  export class Prism extends Component<SyntaxHighlighterProps> {}
  export class Light extends Component<SyntaxHighlighterProps> {}

  const SyntaxHighlighter: React.FC<SyntaxHighlighterProps>
  export default SyntaxHighlighter
}

declare module 'react-syntax-highlighter/dist/cjs/styles/prism' {
  export const oneLight: Record<string, any>
  export const oneDark: Record<string, any>
  export const atomDark: Record<string, any>
  export const dracula: Record<string, any>
  export const okaidia: Record<string, any>
  export const tomorrow: Record<string, any>
}
