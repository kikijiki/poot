import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'poot',
  tagline:
    'Backend-neutral graph IR, automatic kernel fusion, and async capture/replay for LLM inference.',
  favicon: 'img/favicon.ico',

  // Improve compatibility with the upcoming Docusaurus v4.
  future: {
    v4: true,
  },

  // Production deployment root.
  url: process.env.DOCS_SITE_URL ?? 'https://kikijiki.github.io',
  baseUrl: process.env.DOCS_BASE_URL ?? '/',
  trailingSlash: false,

  organizationName: 'kikijiki',
  projectName: 'poot',

  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  markdown: {
    // Parse .md as CommonMark (not MDX) so migrated GitHub markdown with bare
    // `<T>`, `=`, and `{` in prose is treated as literal text. .mdx still gets MDX.
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/kikijiki/poot/tree/master/website/',
        },
        // The website is the user-facing manual only (getting started, architecture, usage, examples).
        // Internal project tracking is kept outside this repo.
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: [
    // Offline / local search (no Algolia, no network).
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        indexDocs: true,
        indexBlog: false,
        docsRouteBasePath: '/docs',
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      },
    ],
  ],

  plugins: [
    // Generate llms.txt / llms-full.txt and per-page markdown for AI consumers.
    [
      'docusaurus-plugin-llms',
      {
        generateLLMsTxt: true,
        generateLLMsFullTxt: true,
        docsDir: 'docs',
        includeBlog: false,
        title: 'poot',
        description:
          'A from-scratch LLM inference engine built on a backend-neutral tensor-op graph IR with automatic kernel fusion and async graph capture and replay.',
      },
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'poot',
      logo: {
        alt: 'poot logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docs',
          position: 'left',
          label: 'Docs',
        },
        {
          type: 'doc',
          docId: 'develop/index',
          position: 'left',
          label: 'Develop',
        },
        {
          type: 'doc',
          docId: 'serve/index',
          position: 'left',
          label: 'Serve',
        },
        {
          type: 'doc',
          docId: 'architecture/index',
          position: 'left',
          label: 'Architecture',
        },
        {
          href: 'https://github.com/kikijiki/poot',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Develop', to: '/docs/develop'},
            {label: 'Serve', to: '/docs/serve'},
            {label: 'Architecture', to: '/docs/architecture'},
            {label: 'Feature matrix', to: '/docs/reference/feature-matrix'},
            {label: 'FAQ', to: '/docs/reference/faq'},
          ],
        },
        {
          title: 'Project',
          items: [
            {label: 'GitHub', href: 'https://github.com/kikijiki/poot'},
          ],
        },
      ],
      copyright: `Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['rust', 'bash', 'toml', 'nix', 'diff', 'json'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
