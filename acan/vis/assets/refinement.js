/* Native graph navigation, zoom and fullscreen; no model changes. */
(function () {
  const bound = new WeakSet();
  const buttons = new WeakSet();
  function current() {
    const el = document.getElementById('org-chart');
    return el && el._cyreg && el._cyreg.cy;
  }
  function fit() {
    const cy = current();
    if (!cy || !cy.nodes().length) return;
    cy.resize();
    if(cy.nodes('.local-root').length) {
      const columns=cy.width()<600?2:3;
      cy.batch(()=>{
        cy.nodes('.local-root').position({x:(columns-1)*87.5,y:20});
        cy.nodes('.local-group').forEach((node,index)=>node.position({x:(index%columns)*175,y:140+Math.floor(index/columns)*115}));
      });
    }
    cy.fit(cy.elements(), 35);
    if (cy.zoom() > 1.2) cy.zoom({level: 1.2, renderedPosition: {x:cy.width()/2,y:cy.height()/2}});
  }
  function bind() {
    const cy = current();
    if (cy && !bound.has(cy)) {
      bound.add(cy);
      cy.on('dbltap', 'node', function (event) {
        const node = event.target;
        const child = node.data('child_id');
        if (child !== undefined || !node.id().includes('_G')) {
          window.dash_clientside.set_props('graph-focus', {value:child !== undefined ? child : Number(node.id())});
          window.dash_clientside.set_props('graph-view', {value:'full'});
        } else {
          window.dash_clientside.set_props('refine-request', {data:{node_id:node.id(),nonce:Date.now()}});
        }
      });
      cy.on('layoutstop', function () { if(cy.nodes('.local-root').length) fit(); });
      const panel = document.getElementById('org-chart');
      new ResizeObserver(() => { cy.resize(); if(cy.nodes('.local-root').length) fit(); }).observe(panel);
    }
    const actions = {
      'graph-fit': fit,
      'graph-zoom-in': () => zoom(1.35),
      'graph-zoom-out': () => zoom(1/1.35),
      'graph-fullscreen': async () => {
        const panel = document.getElementById('graph-panel');
        if (document.fullscreenElement) await document.exitFullscreen();
        else if (panel.classList.contains('graph-fullscreen')) panel.classList.remove('graph-fullscreen');
        else {
          try { await panel.requestFullscreen(); }
          catch (_) { panel.classList.add('graph-fullscreen'); }
        }
        setTimeout(fit,100);
      }
    };
    Object.entries(actions).forEach(([id,action]) => {
      const button = document.getElementById(id);
      if (button && !buttons.has(button)) { buttons.add(button); button.addEventListener('click',action); }
    });
  }
  function zoom(factor) {
    const cy = current();
    if(cy) cy.zoom({level:Math.max(cy.minZoom(),Math.min(cy.maxZoom(),cy.zoom()*factor)),
      renderedPosition:{x:cy.width()/2,y:cy.height()/2}});
  }
  document.addEventListener('fullscreenchange', () => setTimeout(fit,100));
  document.addEventListener('keydown', event => {
    if(event.key==='Escape') {document.getElementById('graph-panel')?.classList.remove('graph-fullscreen');setTimeout(fit,100);}
  });
  new MutationObserver(bind).observe(document.documentElement,{childList:true,subtree:true});
  window.addEventListener('load',bind);
})();
