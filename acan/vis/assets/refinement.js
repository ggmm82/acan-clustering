/* Attach a native Cytoscape double-tap handler once per mounted graph. */
(function () {
  const bound = new WeakSet();
  function bind() {
    const element = document.getElementById('org-chart');
    const cy = element && element._cyreg && element._cyreg.cy;
    if (!cy || bound.has(cy)) return;
    bound.add(cy);
    cy.on('dbltap', 'node', function (event) {
      window.dash_clientside.set_props('refine-request', {
        data: {node_id: event.target.id(), nonce: Date.now()}
      });
    });
  }
  new MutationObserver(bind).observe(document.documentElement, {childList: true, subtree: true});
  window.addEventListener('load', bind);
})();
