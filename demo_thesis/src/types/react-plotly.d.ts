declare module 'react-plotly.js' {
  import type { Component, CSSProperties } from 'react';
  import type Plotly from 'plotly.js';

  interface PlotParams {
    data: Plotly.Data[];
    layout?: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
    frames?: Plotly.Frame[];
    style?: CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (figure: Record<string, unknown>, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: Record<string, unknown>, graphDiv: HTMLElement) => void;
    onPurge?: (figure: Record<string, unknown>, graphDiv: HTMLElement) => void;
    onHover?: (event: Plotly.PlotHoverEvent) => void;
    onClick?: (event: Plotly.PlotMouseEvent) => void;
    revision?: number;
  }

  class PlotComponent extends Component<PlotParams> {}
  export default PlotComponent;
}
