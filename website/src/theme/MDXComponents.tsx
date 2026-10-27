import type {ComponentProps, ReactNode} from 'react';
import MDXComponents from '@theme-original/MDXComponents';

// Every markdown table sits in a frame that scrolls when a cell cannot wrap (see `.table-scroll` in custom.css).
function Table(props: ComponentProps<'table'>): ReactNode {
  return (
    <div className="table-scroll">
      <table {...props} />
    </div>
  );
}

export default {
  ...MDXComponents,
  table: Table,
};
