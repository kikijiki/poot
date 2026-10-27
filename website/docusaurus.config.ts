import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const baseUrl = process.env.DOCS_BASE_URL ?? '/';

const config: Config = {
  title: 'poot',
  tagline:
    'Backend-neutral graph IR, automatic kernel fusion, and async capture/replay for LLM inference.',
  favicon: 'img/favicon.ico',
  headTags: [
    // Crisp vector favicon for browsers that support it; favicon.ico stays as the fallback.
    {
      tagName: 'link',
      attributes: {rel: 'icon', type: 'image/svg+xml', href: `${baseUrl}img/favicon.svg`},
    },
  ],

  // Improve compatibility with the upcoming Docusaurus v4.
  future: {
    v4: true,
  },

  // Production deployment root.
  url: process.env.DOCS_SITE_URL ?? 'https://kikijiki.github.io',
  baseUrl,
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
          // The navbar lists the top-level sections; breadcrumbs would repeat it.
          breadcrumbs: false,
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
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'poot',
      logo: {
        alt: 'poot logo',
        src: 'img/logo.svg',
        srcDark: 'img/logo-dark.svg',
      },
      // One entry per top-level section, in position order (`bun scripts/check-navbar.mjs` checks it): a folder with
      // a _category_.json is a `docSidebar` item, a single page is a `doc` item.
      items: [
        {type: 'docSidebar', sidebarId: 'develop', position: 'left', label: 'Develop'},
        {type: 'docSidebar', sidebarId: 'serve', position: 'left', label: 'Serving'},
        {type: 'docSidebar', sidebarId: 'architecture', position: 'left', label: 'Architecture'},
        {type: 'doc', docId: 'performance/index', position: 'left', label: 'Performance'},
        {type: 'doc', docId: 'status', position: 'left', label: 'Status'},
        {
          href: 'https://github.com/kikijiki/poot',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['rust', 'bash', 'toml', 'nix', 'diff', 'json', 'llvm', 'nasm'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
