/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brain: {
          dark: '#0a0e1a',
          navy: '#0d1529',
          panel: '#131a2e',
          border: '#1e2a4a',
          accent: '#00d4ff',
          purple: '#8b5cf6',
          pink: '#ec4899',
          green: '#10b981',
          amber: '#f59e0b',
          glow: '#00d4ff33',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'float': 'float 3s ease-in-out infinite',
        'scan-line': 'scanLine 2s linear infinite',
      },
      keyframes: {
        pulseGlow: {
          '0%, 100%': { opacity: '0.4', transform: 'scale(1)' },
          '50%': { opacity: '1', transform: 'scale(1.05)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        scanLine: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      }
    },
  },
  plugins: [],
}
