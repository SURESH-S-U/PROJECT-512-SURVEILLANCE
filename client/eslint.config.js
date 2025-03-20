import js from '@eslint/js';
import reactRefresh from 'eslint-plugin-react-refresh';
import reactHooks from 'eslint-plugin-react-hooks';
import react from 'eslint-plugin-react';

export default [
  js.configs.recommended,
  {
    files: ['**/*.{js,jsx}'],
    plugins: {
      'react-refresh': reactRefresh,
      'react-hooks': reactHooks,
      'react': react
    },
    rules: {
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
      'react/react-in-jsx-scope': 'off',
      'react/prop-types': 'off'
    },
    settings: {
      react: {
        version: 'detect'
      }
    }
  }
];
