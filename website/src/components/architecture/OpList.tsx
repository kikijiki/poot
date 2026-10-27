import React from "react";
import { CATEGORIES, type Field, type Op } from "./ops";
import styles from "./OpList.module.css";

// `Reduce(op, axis, keepdim)` -> the op name and its attribute params; `A / B` names two ops that share one entry.
function splitName(full: string): { name: string; params?: string } {
  const m = full.match(/^([^(]+)\((.*)\)$/);
  return m ? { name: m[1], params: `(${m[2]})` } : { name: full };
}

const COLUMNS = ["label", "field", "shape", "meaning"] as const;

// A field's name and shape never break (nowrap cells); only the meaning wraps.
function FieldCells({ field }: { field: Field }): React.ReactElement {
  return (
    <>
      <td className={styles.fieldName}>
        <code>{field.name}</code>
      </td>
      <td className={styles.shape}>{field.shape}</td>
      <td className={styles.meaning}>{field.meaning}</td>
    </>
  );
}

// One table per op: a header spanning every column, an Inputs label cell spanning the input rows, the Output row, and
// a notes row spanning every column. All tables share one colgroup, so columns line up down the page.
function OpTable({ op }: { op: Op }): React.ReactElement {
  const { name, params } = splitName(op.name);
  return (
    <div className="table-scroll">
      <table className={styles.table}>
        <colgroup>
          {COLUMNS.map((c) => (
            <col className={styles[`col_${c}`]} key={c} />
          ))}
        </colgroup>
        <thead>
          <tr>
            <th colSpan={COLUMNS.length} scope="colgroup" className={styles.title}>
              <code className={styles.name}>{name}</code>
              {params && <span className={styles.params}>{params}</span>}
            </th>
          </tr>
        </thead>
        <tbody>
          {op.inputs.map((input, i) => (
            <tr key={input.name}>
              {i === 0 && (
                <th rowSpan={op.inputs.length} scope="rowgroup" className={styles.label}>
                  Inputs
                </th>
              )}
              <FieldCells field={input} />
            </tr>
          ))}
          <tr>
            <th scope="row" className={styles.label}>
              Output
            </th>
            <FieldCells field={op.output} />
          </tr>
          <tr>
            <td colSpan={COLUMNS.length} className={styles.notes}>
              {op.notes}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

// One category of the graph IR op reference: one table per op, all on the same fixed columns.
export default function OpList({ category }: { category: string }): React.ReactElement {
  const found = CATEGORIES.find((c) => c.id === category);
  if (!found) throw new Error(`unknown op category: ${category}`);
  return (
    <div className={styles.ops}>
      {found.ops.map((op) => (
        <OpTable op={op} key={op.name} />
      ))}
    </div>
  );
}
