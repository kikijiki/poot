import React from "react";
import Link from "@docusaurus/Link";
import data from "@site/src/data/status.json";
import { Icon } from "./Icon";
import styles from "./status.module.css";
import { badgeLabel, type StatusData } from "./types";

const board = data as StatusData;

export default function StatusBoard() {
  return (
    <div className={styles.root}>
      <div className={styles.columns}>
        {board.columns.map((column) => (
          <section key={column.id} className={styles.column} aria-labelledby={`status-${column.id}`}>
            <h2 id={`status-${column.id}`} className={styles.columnTitle}>
              {column.title}
            </h2>
            <ul className={styles.cards}>
              {board.cards
                .filter((card) => card.column === column.id)
                .map((card) => (
                  <li key={card.id}>
                    <Link to={card.link} className={styles.card}>
                      <span className={styles.icon}>
                        <Icon name={card.icon} />
                      </span>
                      <span className={styles.body}>
                        <span className={styles.title}>{card.title}</span>
                        <span className={styles.line}>{card.line}</span>
                        {card.groups?.map((group) => (
                          <span key={group.label} className={styles.group}>
                            <span className={styles.groupLabel}>{group.label}</span>
                            {group.items.map((item) => item.name).join(", ")}
                          </span>
                        ))}
                      </span>
                      <span className={`${styles.badge} ${styles[`badge-${card.badge}`]}`}>{badgeLabel[card.badge]}</span>
                    </Link>
                  </li>
                ))}
            </ul>
          </section>
        ))}
      </div>
      <p className={styles.later}>
        <strong>Later.</strong> {board.later}
      </p>
      <div className={styles.limits}>
        <h2 className={styles.limitsTitle}>Known limitations</h2>
        <ul>
          {board.limitations.map((text) => (
            <li key={text}>{text}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
