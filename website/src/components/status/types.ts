export type ColumnId = "works" | "building" | "next";
export type Badge = "works" | "partial" | "in-progress" | "experimental" | "planned";

// An item names something concrete; `ref` is the row of the supported-models page that proves it (a family key or a format name).
export type Item = { name: string; ref?: string };
export type Group = { label: string; items: Item[] };
export type Card = { id: string; column: ColumnId; icon: string; title: string; line: string; badge: Badge; link: string; groups?: Group[] };
export type Column = { id: ColumnId; title: string };
export type StatusData = { columns: Column[]; later: string; limitations: string[]; cards: Card[] };

export const badgeLabel: Record<Badge, string> = {
  works: "Works",
  partial: "Partial",
  "in-progress": "In progress",
  experimental: "Experimental",
  planned: "Planned",
};
