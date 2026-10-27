import React from "react";

// One 24px stroke icon per card, drawn in currentColor.
const paths: Record<string, string> = {
  server: "M4 5h16v5H4zM4 14h16v5H4zM7.5 7.5h.01M7.5 16.5h.01",
  layers: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5",
  braces: "M9 4C6 4 7 9 5 12c2 3 1 8 4 8M15 4c3 0 2 5 4 8-2 3-1 8-4 8",
  bolt: "M13 3L5 14h6l-1 7 8-11h-6z",
  puzzle: "M5 8h4a2 2 0 114 0h4v4a2 2 0 110 4v4H5v-5a2 2 0 100-4z",
  image: "M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4M15 9h.01",
  grid: "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
  cube: "M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5",
  file: "M6 3h8l4 4v14H6zM14 3v4h4M9 12h6M9 16h6",
  chip: "M7 7h10v10H7zM10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4",
  chart: "M4 20V4M4 20h16M8 16v-4M12 16V8M16 16v-6",
};

export function Icon({ name }: { name: string }) {
  const d = paths[name];
  if (!d) throw new Error(`unknown status icon: ${name}`);
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  );
}

export const iconNames = Object.keys(paths);
