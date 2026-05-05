import type { SVGProps } from 'react';

type IconProps = SVGProps<SVGSVGElement> & { className?: string };

function strokeIcon(props: IconProps) {
  const { className, ...rest } = props;
  return {
    className,
    ...rest,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.5,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    width: '1em',
    height: '1em',
  };
}

/** Brain / neuroscience — nav Brain tab, cortical signals */
export function IconBrain(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M10 5C8 5 6.5 6.2 6 8c-.8-.3-1.7 0-2.3.7-.8.9-.8 2.3 0 3.2.3.4.8.6 1.3.7-.2 1.9 1 3.6 2.8 4.2.5.2 1 .3 1.5.3 1 0 2-.4 2.7-1 .7.6 1.7 1 2.7 1 .5 0 1-.1 1.5-.3 1.8-.6 3-2.3 2.8-4.2.5-.1 1-.3 1.3-.7.8-.9.8-2.3 0-3.2-.6-.7-1.5-1-2.3-.7C17.5 6.2 16 5 14 5s-3.5 1.2-4 3c-.5-1.8-2-3-4-3z" />
      <path d="M12 11v5M10 13h4" />
    </svg>
  );
}

/** Pipeline / flow — circuit-style nodes */
export function IconPipeline(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <rect x="3" y="4" width="6" height="6" rx="1.25" />
      <rect x="15" y="4" width="6" height="6" rx="1.25" />
      <rect x="9" y="14" width="6" height="6" rx="1.25" />
      <path d="M9 7h3M18 10v3M15 17h-3M12 14V11M12 11l3-4" />
    </svg>
  );
}

/** Explorer — compass */
export function IconExplore(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <circle cx="12" cy="12" r="9" />
      <path d="M14.5 9.5l-3 7-1.5-3.5L9.5 11l5-1.5z" />
    </svg>
  );
}

/** Challenge — trophy */
export function IconChallenge(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M8 21h8M12 17v4M7 8h10l1 3a4.5 4.5 0 01-9 0l1-3z" />
      <path d="M7 11H5a2 2 0 100 4h2M17 11h2a2 2 0 010 4h-2" />
    </svg>
  );
}

export function IconHome(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M4 10.5 12 4l8 6.5V20a1 1 0 01-1 1h-5v-6H10v6H5a1 1 0 01-1-1v-9.5z" />
    </svg>
  );
}

export function IconFilm(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <rect x="4" y="5" width="16" height="14" rx="2" />
      <path d="M8 5v14M16 5v14M4 9h16M4 15h16" />
    </svg>
  );
}

export function IconChevronDown(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}

export function IconChevronRight(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M9 6l6 6-6 6" />
    </svg>
  );
}

export function IconCheck(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M5 13l4 4L19 7" />
    </svg>
  );
}

export function IconSkip(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M5 6v12l7-6-7-6z" />
      <path d="M13 6v12l7-6-7-6z" />
    </svg>
  );
}

export function IconSearch(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <circle cx="11" cy="11" r="6.5" />
      <path d="M16 16l4.5 4.5" />
    </svg>
  );
}

export function IconChart(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M5 19h14M7 15v4M11 11v8M15 7v12M19 3v16" />
    </svg>
  );
}

/** CLIP / semantic space — atom symbol */
export function IconAtom(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <circle cx="12" cy="12" r="2.25" />
      <ellipse cx="12" cy="12" rx="9" ry="4.25" />
      <ellipse cx="12" cy="12" rx="9" ry="4.25" transform="rotate(60 12 12)" />
      <ellipse cx="12" cy="12" rx="9" ry="4.25" transform="rotate(-60 12 12)" />
    </svg>
  );
}

export function IconLayers(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M12 4l9 5-9 5-9-5 9-5zM3 10l9 5 9-5M3 14l9 5 9-5" />
    </svg>
  );
}

export function IconGauge(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M12 14a2 2 0 100-4 2 2 0 000 4z" />
      <path d="M12 14v3" />
      <path d="M4 13a8 8 0 0116 0" />
      <path d="M6 17l2-3M18 17l-2-3" />
    </svg>
  );
}

export function IconDownload(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M12 4v11m0 0l4-4m-4 4l-4-4M5 19h14" />
    </svg>
  );
}

export function IconEye(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M2 12s4.5-7 10-7 10 7 10 7-4.5 7-10 7S2 12 2 12z" />
      <circle cx="12" cy="12" r="3.25" />
    </svg>
  );
}

export function IconSparkles(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" />
      <path d="M12 8l1.2 3.8h4l-3.2 2.4 1.2 3.8L12 15.8 8.8 18l1.2-3.8L6.8 11.8h4L12 8z" />
    </svg>
  );
}

export function IconPlay(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M9 7l10 5-10 5V7z" />
    </svg>
  );
}

export function IconArrowLeft(props: IconProps) {
  const p = strokeIcon(props);
  return (
    <svg {...p}>
      <path d="M19 12H5M12 19l-7-7 7-7" />
    </svg>
  );
}

export const iconMap = {
  brain: IconBrain,
  pipeline: IconPipeline,
  explore: IconExplore,
  challenge: IconChallenge,
  home: IconHome,
  film: IconFilm,
  chevronDown: IconChevronDown,
  chevronRight: IconChevronRight,
  check: IconCheck,
  skip: IconSkip,
  search: IconSearch,
  chart: IconChart,
  atom: IconAtom,
  layers: IconLayers,
  gauge: IconGauge,
  download: IconDownload,
  eye: IconEye,
  sparkles: IconSparkles,
  play: IconPlay,
  arrowLeft: IconArrowLeft,
} as const;

export type IconName = keyof typeof iconMap;
