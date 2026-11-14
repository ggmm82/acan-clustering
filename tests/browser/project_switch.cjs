const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const net=require('node:net');
(async()=>{
 const port=await new Promise(resolve=>{const s=net.createServer();s.listen(0,'127.0.0.1',()=>{const p=s.address().port;s.close(()=>resolve(p));});});
 const server=spawn(process.env.PYTHON||'python',['tests/browser/project_switch_fixture.py',String(port)],{stdio:['ignore','ignore','pipe']});
 let logs='';server.stderr.on('data',x=>logs+=x);
 let browser;
 try{
  const url=`http://127.0.0.1:${port}`;
  let ready=false;
  for(let i=0;i<60;i++){try{if((await fetch(url+'/healthz')).ok){ready=true;break;}}catch(_){}await new Promise(r=>setTimeout(r,250));}
  assert(ready,logs);
  browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{}),args:['--no-sandbox']});
  const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);await page.getByText('Generativo',{exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#lm-connection').innerText.includes('Worker collegato'));
  for(const n of [1,2,1,3]){
   await page.locator('#lm-job').click();
   await page.getByText(`Project ${n} · paused · passo 1500`,{exact:true}).last().click();
   await page.waitForFunction(number=>document.querySelector('#lm-loss .js-plotly-plot')?.data?.[0]?.y?.[0]===number,n);
  }
  await page.evaluate(async()=>{await Plotly.relayout(document.querySelector('#lm-loss .js-plotly-plot'),{'xaxis.range':[200,400]});});
  await page.evaluate(()=>{window.fixtureAfterPlot=new Promise(resolve=>document.querySelector('#lm-loss .js-plotly-plot').once('plotly_afterplot',()=>resolve()));});
  await page.locator('#lm-refresh').click();
  await page.evaluate(async()=>{await window.fixtureAfterPlot;});

  await page.waitForFunction(()=>document.querySelector('#lm-loss .js-plotly-plot')?.layout?.xaxis?.range?.[0]===200);
  await page.evaluate(()=>{window.dash_clientside.set_props('lm-history-mode',{value:'window'});window.dash_clientside.set_props('lm-history-start',{value:1000});});
  await page.waitForFunction(()=>document.querySelector('#lm-loss .js-plotly-plot')?.data?.[0]?.x?.[0]===1000);
  const calls=await(await fetch(url+'/fixture/calls')).json();
  assert(calls.some(p=>p==='/jobs/job-2/view'));
  assert(calls.filter(p=>p.startsWith('/state')).every(p=>p.includes('summary=1')));
  await page.waitForFunction(()=>document.querySelector('#lm-decision')?.innerText.includes('Decisione sospesa'));
  await page.waitForFunction(()=>document.querySelector('#lm-small-status')?.innerText.includes('Nessuna prova'));
  assert.deepEqual(errors,[]);
  console.log('Persistent projects: repeated selection, matching chart, zoom and exact history window passed.');
 }finally{if(browser)await browser.close();server.kill('SIGTERM');}
})().catch(e=>{console.error(e);process.exit(1)});
