/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // ── Premium scientific graphite palette ──
        // Old brain-* tokens kept for backward compat but values are now
        // neutralized (R≈G≈B) — no more purple/blue tint.
        brain: {
          dark:   '#0a0a0b',
          navy:   '#101011',
          panel:  '#16171a',
          border: '#26272b',
          accent: '#4d7cff',
          purple: '#7c6ff7',
          pink:   '#e8639a',
          green:  '#34d399',
          amber:  '#fbbf24',
          glow:   '#4d7cff1a',
        },
        // ── Neutral graphite surface elevation system ──
        // RGB channels are intentionally close so dark surfaces read as true
        // graphite, not bluish/purple. Lift through the scale is luminance,
        // never hue.
        surface: {
          base:     '#0a0a0b',
          raised:   '#101012',
          elevated: '#16171a',
          overlay:  '#1d1e22',
          hover:    '#22232a',
          active:   '#282a31',
        },
        // ── Neutral border hierarchy ──
        border: {
          subtle:   '#1f2023',
          DEFAULT:  '#26272b',
          emphasis: '#2f3036',
          strong:   '#3d3d44',
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
