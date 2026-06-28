
(function () {
  window.golgi_export_plot = function (tileId, format) {
    var root = document.getElementById(tileId);
    if (!root) {
      console.warn('[golgi-export] tile not found:', tileId);
      return;
    }
    var gd = root.querySelector('.js-plotly-plot');
    if (!gd) {
      console.warn(
        '[golgi-export] no plotly div in tile:', tileId
      );
      return;
    }
    if (!window.Plotly) {
      console.warn('[golgi-export] Plotly global missing');
      return;
    }
    var opts = {
      format: (format || 'png'),
      filename: tileId,
      scale: 2,
    };
    try {
      window.Plotly.downloadImage(gd, opts);
    } catch (err) {
      console.warn('[golgi-export] downloadImage failed', err);
    }
  };
})();
