import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  // Build output is not linted.
  { ignores: ['dist'] },
  // Source files: TS + React.
  {
    files: ['src/**/*.{ts,tsx}'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // Fetch-on-mount and initial-value sync patterns are pervasive in this
      // codebase; refactoring them is tracked separately, keep as warnings.
      'react-hooks/set-state-in-effect': 'warn',
      // HMR boundaries: warn only, matches the Vite react-ts template.
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
      // The codebase intentionally keeps some unused params (interface
      // implementations); `_`-prefixed names are the escape hatch.
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_', caughtErrors: 'none' },
      ],
      // Legacy API-layer and ambient declarations still use `any`; keep
      // visible as warnings instead of blocking CI. Tighten over time.
      '@typescript-eslint/no-explicit-any': 'warn',
      // Empty catch blocks are allowed (API calls often swallow errors);
      // other empty blocks stay errors.
      'no-empty': ['error', { allowEmptyCatch: true }],
    },
  },
)
