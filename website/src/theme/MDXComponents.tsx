import {Children, isValidElement} from 'react';
import type {ComponentProps, ReactElement, ReactNode} from 'react';
import MDXComponents from '@theme-original/MDXComponents';

// The text of the last header cell of a markdown table (`thead > tr > th`), or undefined if the shape is unexpected.
function lastHeader(table: ReactNode): string | undefined {
  const elements = (node: ReactNode): ReactElement<{children?: ReactNode}>[] =>
    Children.toArray(node).filter(isValidElement) as ReactElement<{children?: ReactNode}>[];
  const head = elements(table).find((e) => e.type === 'thead');
  const row = head && elements(head.props.children)[0];
  const cell = row && elements(row.props.children).at(-1);
  return typeof cell?.props.children === 'string' ? cell.props.children : undefined;
}

// Every markdown table sits in a frame that scrolls when a cell cannot wrap (see `.table-scroll` in custom.css). A table
// whose last column is Notes wraps only that column: the others keep their tokens whole and the frame scrolls instead.
function Table(props: ComponentProps<'table'>): ReactNode {
  const notesLast = lastHeader(props.children) === 'Notes';
  return (
    <div className="table-scroll">
      <table {...props} className={notesLast ? 'table-notes-last' : undefined} />
    </div>
  );
}

export default {
  ...MDXComponents,
  table: Table,
};
