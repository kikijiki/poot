import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

const TAGLINE =
  'Backend-neutral graph IR, automatic kernel fusion, and async capture/replay for LLM inference.';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  const logo = useBaseUrl('/img/logo-dark.svg');
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <img className={styles.heroLogo} src={logo} alt="" width={112} height={112} />
        <Heading as="h1" className={clsx('hero__title', styles.heroTitle)}>
          {siteConfig.title}
        </Heading>
        <p className={styles.heroQuote}>I-I need to, I need to COMPOOOOT</p>
        <p className="hero__subtitle">{TAGLINE}</p>
        <p className={styles.heroBackends}>
          Runs on <Link to="/docs/serve/choosing-a-backend#supported-backends">wgpu, Vulkan, ROCm and PTX</Link>.
        </p>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="Backend-neutral graph IR, automatic kernel fusion, and async capture/replay for LLM inference.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
