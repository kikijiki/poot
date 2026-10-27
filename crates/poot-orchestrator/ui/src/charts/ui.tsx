import React from 'react';
import styles from './styles.module.css';

export function ToggleGroup<T extends string>(props: {
  options: {value: T; label: string}[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className={styles.toggleGroup}>
      {props.options.map((o) => (
        <button
          key={o.value}
          type="button"
          className={o.value === props.value ? styles.active : undefined}
          onClick={() => props.onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Select<T extends string>(props: {
  label: string;
  value: T;
  options: {value: T; label: string}[];
  onChange: (v: T) => void;
}) {
  return (
    <div className={styles.control}>
      <label>{props.label}</label>
      <select
        value={props.value}
        onChange={(e) => props.onChange(e.target.value as T)}>
        {props.options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export function ControlWrap(props: {label: string; children: React.ReactNode}) {
  return (
    <div className={styles.control}>
      <label>{props.label}</label>
      {props.children}
    </div>
  );
}

export function ChartCard(props: {
  title: string;
  sub?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className={styles.chartCard}>
      <div className={styles.chartTitle}>{props.title}</div>
      {props.sub ? <div className={styles.chartSub}>{props.sub}</div> : null}
      {props.children}
    </div>
  );
}

// The legend of a chart, in the page above the plot: it wraps to as many lines as the width needs, so it can
// never overlap an axis title (a legend drawn inside the SVG has a fixed height and does).
export function ChartLegend(props: {items: {label: string; color: string; dashed?: boolean}[]}) {
  return (
    <div className={styles.legend}>
      {props.items.map((item) => (
        <span className={styles.legendItem} key={item.label}>
          <span
            className={styles.legendSwatch}
            style={{
              background: item.dashed
                ? `repeating-linear-gradient(90deg, ${item.color} 0 5px, transparent 5px 8px)`
                : item.color,
            }}
          />
          {item.label}
        </span>
      ))}
    </div>
  );
}

export function Empty(props: {children: React.ReactNode}) {
  return <div className={styles.empty}>{props.children}</div>;
}
