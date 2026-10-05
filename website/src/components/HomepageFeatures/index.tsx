import type { ReactNode } from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  href: string;
  Icon: React.ComponentType<React.ComponentProps<"svg">>;
  description: ReactNode;
};

function DevelopIcon(props: React.ComponentProps<"svg">) {
  return (
    <svg
      viewBox="0 0 64 64"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      {...props}
    >
      <rect x="8" y="14" width="48" height="36" rx="2" />
      <path d="M16 24h32M16 32h20M16 40h28" strokeLinecap="round" />
      <path d="M44 38l8 6-8 6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ServeIcon(props: React.ComponentProps<"svg">) {
  return (
    <svg
      viewBox="0 0 64 64"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      {...props}
    >
      <rect x="10" y="12" width="44" height="14" rx="2" />
      <rect x="10" y="38" width="44" height="14" rx="2" />
      <circle cx="20" cy="19" r="2" fill="currentColor" stroke="none" />
      <circle cx="20" cy="45" r="2" fill="currentColor" stroke="none" />
      <path d="M32 26v12M40 26v12M48 26v12" strokeLinecap="round" />
    </svg>
  );
}

function ArchitectureIcon(props: React.ComponentProps<"svg">) {
  return (
    <svg
      viewBox="0 0 64 64"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      {...props}
    >
      <circle cx="32" cy="12" r="6" />
      <circle cx="14" cy="48" r="6" />
      <circle cx="50" cy="48" r="6" />
      <path d="M32 18v14M20 42l8-12M44 42l-8-12" strokeLinecap="round" />
      <rect x="24" y="28" width="16" height="10" rx="1" />
    </svg>
  );
}

const FeatureList: FeatureItem[] = [
  {
    title: "Develop",
    href: "/docs/develop",
    Icon: DevelopIcon,
    description: (
      <>
        Build the workspace, run graphs and kernels as a compute engine, or use
        the LLM <code>Runner</code>{" "}
        API and embed crates in your own project.
      </>
    ),
  },
  {
    title: "Serve",
    href: "/docs/serve",
    Icon: ServeIcon,
    description: (
      <>
        Run <code>poot-serve</code> for an OpenAI-compatible HTTP API that
        bounds requests and serves embeddings, reranking and LoRA
        administration. Generation is not available yet (poot is being refactored; planned to return).
      </>
    ),
  },
  {
    title: "Architecture",
    href: "/docs/architecture",
    Icon: ArchitectureIcon,
    description: (
      <>
        Read how tracing, fusion, capture/replay, and backend lowering fit
        together on a fine-grained graph IR.
      </>
    ),
  },
];

function Feature({ title, href, Icon, description }: FeatureItem) {
  return (
    <div className={clsx("col col--4")}>
      <Link to={href} className={styles.featureLink}>
        <div className="text--center">
          <Icon className={styles.featureIcon} role="img" aria-label={title} />
        </div>
        <div className="text--center padding-horiz--md">
          <Heading as="h3">{title}</Heading>
          <p>{description}</p>
        </div>
      </Link>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props) => (
            <Feature key={props.title} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
