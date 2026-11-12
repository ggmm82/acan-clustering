const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const fs=require('node:fs/promises'),os=require('node:os'),path=require('node:path');
(async()=>{
  const temp=await fs.mkdtemp(path.join(os.tmpdir(),'acan-graph-'));
  const file=path.join(temp,'model.json');
  const json=execFileSync(process.env.PYTHON||'python',['-c',"import runpy; print(runpy.run_path('tests/test_graph_view.py')['fixture']().to_json())"],{encoding:'utf8'});
  await fs.writeFile(file,json);const root=JSON.parse(json).rootId;
  const child=Object.values(JSON.parse(json).clusterModel[root].children)[0];
  const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{}),args:['--no-sandbox']});
  try{
    for(const width of [550,1440]){
      const page=await browser.newPage({viewport:{width,height:1000}});const errors=[];
      page.on('pageerror',e=>errors.push(e.message));
      await page.goto(process.env.ACAN_TEST_URL||'http://127.0.0.1:4000');
      await page.getByText('Model',{exact:true}).click();
      await page.locator('#upload-model input[type=file]').setInputFiles(file);
      await page.waitForFunction(()=>document.getElementById('org-chart')?._cyreg?.cy.nodes('.local-group').length===12);
      await page.locator('#graph-window-summary').filter({hasText:'15 prototipi senza assegnazioni hard'}).waitFor();
      await page.locator('#graph-fit').click();
      const metrics=await page.evaluate(()=>{let cy=document.getElementById('org-chart')._cyreg.cy,n=cy.nodes('.local-group')[0];return {nodes:cy.nodes().length,width:n.renderedWidth(),font:parseFloat(n.style('font-size'))*cy.zoom(),zoom:cy.zoom()}});
      assert.equal(metrics.nodes,13);assert(metrics.width>=95,JSON.stringify(metrics));assert(metrics.font>=9.5,JSON.stringify(metrics));
      await page.evaluate(()=>window.dash_clientside.set_props('graph-page',{value:2}));
      await page.waitForFunction(root=>!!document.getElementById('org-chart')._cyreg.cy.$id(`${root}_G12`).length,root);
      await page.evaluate(()=>window.dash_clientside.set_props('graph-page',{value:1}));
      await page.waitForFunction(root=>!!document.getElementById('org-chart')._cyreg.cy.$id(`${root}_G0`).length,root);
      await page.evaluate(root=>document.getElementById('org-chart')._cyreg.cy.$id(`${root}_G0`).emit('dbltap'),root);
      await page.waitForFunction(child=>!!document.getElementById('org-chart')._cyreg.cy.$id(`${child}_G0`).length,child);
      await page.locator('#graph-up').click();
      await page.waitForFunction(root=>!!document.getElementById('org-chart')._cyreg.cy.$id(`${root}_G0`).length,root);
      await page.evaluate(()=>window.dash_clientside.set_props('graph-view',{value:'hierarchy'}));
      await page.waitForFunction(()=>document.getElementById('org-chart')._cyreg.cy.nodes().length===2);
      await page.evaluate(child=>document.getElementById('org-chart')._cyreg.cy.$id(String(child)).emit('dbltap'),child);
      await page.waitForFunction(child=>!!document.getElementById('org-chart')._cyreg.cy.$id(`${child}_G0`).length,child);
      await page.locator('#graph-root').click();
      await page.waitForFunction(root=>!!document.getElementById('org-chart')._cyreg.cy.$id(`${root}_G0`).length,root);
      const zoom=await page.evaluate(()=>document.getElementById('org-chart')._cyreg.cy.zoom());
      await page.locator('#graph-zoom-in').click();
      await page.waitForFunction(previous=>document.getElementById('org-chart')._cyreg.cy.zoom()>previous,zoom);
      await page.locator('#graph-fullscreen').click();
      await page.waitForFunction(()=>document.fullscreenElement?.id==='graph-panel'||document.getElementById('graph-panel').classList.contains('graph-fullscreen'));
      await page.locator('#graph-focus').click();
      await page.getByText(`Radice ${root} › G0 › ${child} · 160 righe`,{exact:true}).last().click();
      await page.waitForFunction(child=>!!document.getElementById('org-chart')._cyreg.cy.$id(`${child}_G0`).length,child);
      await page.locator('#graph-root').click();
      await page.waitForFunction(root=>!!document.getElementById('org-chart')._cyreg.cy.$id(`${root}_G0`).length,root);
      await page.locator('#graph-fullscreen').click();
      await page.waitForFunction(()=>!document.fullscreenElement&&!document.getElementById('graph-panel').classList.contains('graph-fullscreen'));
      assert.deepEqual(errors,[]);
      console.log(JSON.stringify({viewport:width,...metrics,navigation:'passed',fullscreen:'passed'}));
      await page.screenshot({path:process.env.GRAPH_SCREENSHOT||path.join(temp,`graph-${width}.png`),fullPage:true});
      await page.close();
    }
  }finally{await browser.close();await fs.rm(temp,{recursive:true,force:true});}
})().catch(e=>{console.error(e);process.exit(1)});
