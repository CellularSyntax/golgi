
(function () {
  // Set the browser-tab title + favicon. trame's module system
  // injects scripts/styles but doesn't expose a hook for raw
  // <link rel="icon"> tags, so we install ours at page load
  // and remove any default favicons the trame template ships.
  function installBranding() {
    if (document.title !== 'GOLGI.IO') document.title = 'GOLGI.IO';
    document.querySelectorAll(
      'link[rel*="icon"]'
    ).forEach(function (el) {
      if (!el.hasAttribute('data-golgi')) el.remove();
    });
    if (!document.querySelector('link[rel="icon"][data-golgi]')) {
      const link = document.createElement('link');
      link.rel = 'icon';
      link.type = 'image/png';
      link.href = 'golgi_static/favicon.png';
      link.setAttribute('data-golgi', '1');
      document.head.appendChild(link);
    }
  }

  // Tailwind Play CDN loader. Set the `tailwind` config global
  // BEFORE the script loads so the CDN picks it up on init —
  // notably `corePlugins.preflight = false` to keep Tailwind's
  // aggressive CSS reset from clobbering the existing Vuetify +
  // custom styles. Also extends `theme.animation` with the
  // `border` keyframe used by the SaaS-style animated-gradient
  // border pattern. Safe to call once: bail if already loaded.
  function installTailwind() {
    if (document.querySelector('script[data-golgi-tailwind]')) {
      return;
    }
    if (!window.tailwind) {
      window.tailwind = {};
    }
    window.tailwind.config = {
      corePlugins: { preflight: false },
      theme: {
        extend: {
          animation: {
            'border': 'border 4s linear infinite',
          },
          keyframes: {
            'border': {
              to: { '--border-angle': '360deg' },
            },
          },
        },
      },
    };
    const tw = document.createElement('script');
    tw.src = 'https://cdn.tailwindcss.com';
    tw.setAttribute('data-golgi-tailwind', '1');
    document.head.appendChild(tw);
  }

  const FILTER_ID = 'loader-recolor';
  const SVG_NS = 'http://www.w3.org/2000/svg';
  // (R, G, B) for the target loader colour, precomputed for #e24b4a.
  const MATRIX_VALUES = (
    '0.114 0 0 0 0.886 ' +
    '0 0.706 0 0 0.294 ' +
    '0 0 0.710 0 0.290 ' +
    '0 0 0 1 0'
  );

  function ensureSvgFilter() {
    if (document.getElementById(FILTER_ID)) return;
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('style',
      'position: absolute; width: 0; height: 0; overflow: hidden;'
    );
    svg.setAttribute('aria-hidden', 'true');
    const filter = document.createElementNS(SVG_NS, 'filter');
    filter.setAttribute('id', FILTER_ID);
    filter.setAttribute('color-interpolation-filters', 'sRGB');
    const m = document.createElementNS(SVG_NS, 'feColorMatrix');
    m.setAttribute('type', 'matrix');
    m.setAttribute('values', MATRIX_VALUES);
    filter.appendChild(m);
    svg.appendChild(filter);
    document.body.appendChild(svg);
  }

  const variants = ['v1', 'v2', 'v3', 'v4', 'v5'];
  let currentIdx = 4;

  function apply(idx) {
    document.querySelectorAll('.loader').forEach(function (el) {
      variants.forEach(function (v) { el.classList.remove(v); });
      el.classList.add(variants[idx]);
    });
    currentIdx = idx;
  }

  function start() {
    installBranding();
    installTailwind();
    ensureSvgFilter();
    if (!document.querySelector('.loader')) {
      // Vue hasn't mounted the lightbox yet — retry shortly.
      // Re-run installBranding in the retry loop so trame's
      // late-loaded HTML can't blow away our favicon/title.
      setTimeout(start, 250);
      return;
    }
    apply(currentIdx);  // sync DOM class with our index
    setInterval(function () {
      let next;
      do {
        next = Math.floor(Math.random() * variants.length);
      } while (next === currentIdx);
      apply(next);
    }, 10000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
