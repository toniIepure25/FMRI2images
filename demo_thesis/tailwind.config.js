/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // ── Premium dark workbench palette ──
        // Old brain-* tokens remapped to refined values for backward compat.
        brain: {
          dark:   '#09090b',   // base background  (was #0a0e1a)
          navy:   '#0f0f14',   // raised section   (was #0d1529)
          panel:  '#15151c',   // elevated card    (was #131a2e)
          border: '#252533',   // subtle border    (was #1e2a4a)
          accent: '#4d7cff',   // restrained blue  (was #00d4ff)
          purple: '#7c6ff7',   // softened purple  (was #8b5cf6)
          pink:   '#e8639a',   // softened pink    (was #ec4899)
          green:  '#34d399',   // muted success    (was #10b981)
          amber:  '#fbbf24',   // muted warning    (was #f59e0b)
          glow:   '#4d7cff1a', // accent ghost     (was #00d4ff33)
        },
        // ── NEW surface elevation system ──
        surface: {
          base:     '#09090b',
          raised:   '#0f0f14',
          elevated: '#15151c',
          overlay:  '#1c1c25',
          hover:    '#1f1f2a',
          active:   '#252533',
        },
        // ── Border hierarchy ──
        border: {
          subtle:   '#1e1e2a',
          DEFAULT:  '#252533',
          emphasis: '#303045',
          strong:   '#3d3d55',
        },
        // ── Single restrained accent ──
        accent: {
          DEFAULT: '#4d7cff',
          muted:   '#3b5fd9',
          soft:    '#4d7cff1a',
          hover:   '#6b94ff',
        },
        // ── Text hierarchy ──
        text: {
          primary:   '#fafafa',
          secondary: '#a1a1aa',
          muted:     '#71717a',
          inverse:   '#09090b',
        },
        // ── Status (muted, research-grade) ──
        status: {
          success: '#34d399',
          warning: '#fbbf24',
          error:   '#f87171',
          info:    '#60a5fa',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'Consolas', 'monospace'],
      },
      borderRadius: {
        '2xl': '0.75rem',
        '3xl': '1rem',
      },
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.875rem' }],
      },
      boxShadow: {
        'surface': '0 1px 2px 0 rgb(0 0 0 / 0.30), 0 1px 0 0 rgb(255 255 255 / 0.02)',
        'surface-lg': '0 4px 12px -2px rgb(0 0 0 / 0.40), 0 2px 4px -1px rgb(0 0 0 / 0.20)',
        'surface-xl': '0 8px 24px -4px rgb(0 0 0 / 0.50)',
      },
    },
  },
  plugins: [],
}
