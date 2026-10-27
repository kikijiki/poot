import type {ReactNode} from 'react';
import Head from '@docusaurus/Head';
import {Redirect} from '@docusaurus/router';
import useBaseUrl from '@docusaurus/useBaseUrl';

// There is no docs landing page: the navbar lists the sections, so /docs goes to the first page a reader needs.
const FIRST_PAGE = '/docs/serve/serving-your-first-model';

export default function DocsRoot(): ReactNode {
  const target = useBaseUrl(FIRST_PAGE);
  return (
    <>
      <Head>
        <meta httpEquiv="refresh" content={`0; url=${target}`} />
        <meta name="robots" content="noindex" />
      </Head>
      <Redirect to={FIRST_PAGE} />
    </>
  );
}
