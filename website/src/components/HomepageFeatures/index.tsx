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
      <path d="M22 20L9 32L22 44" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M42 20L55 32L42 44" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M37 14L27 50" strokeLinecap="round" />
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
      <rect x="10" y="10" width="44" height="12" rx="2" />
      <rect x="10" y="26" width="44" height="12" rx="2" />
      <rect x="10" y="42" width="44" height="12" rx="2" />
      <circle cx="18" cy="16" r="1.6" fill="currentColor" stroke="none" />
      <circle cx="18" cy="32" r="1.6" fill="currentColor" stroke="none" />
      <circle cx="18" cy="48" r="1.6" fill="currentColor" stroke="none" />
      <path d="M30 16h16M30 32h16M30 48h16" strokeLinecap="round" />
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
      <path
        d="M28.7 14.8L17.3 28.2M35.3 14.8L46.7 28.2M17.3 35.8L28.7 49.2M46.7 35.8L35.3 49.2"
        strokeLinecap="round"
      />
      <circle cx="32" cy="11" r="5" />
      <circle cx="14" cy="32" r="5" />
      <circle cx="50" cy="32" r="5" />
      <circle cx="32" cy="53" r="5" />
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
        the LLM <code>Driver</code>{" "}
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
        bounds requests. It serves generation, embeddings, reranking and
        LoRA adapter administration.
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
