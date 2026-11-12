const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const os=require('node:os');
const path=require('node:path');
(async()=>{
 const temporary=await fs.mkdtemp(path.join(os.tmpdir(),'acan-numeric-'));
 const input=path.join(temporary,'classes.csv');
 await fs.writeFile(input,['feature,class',...Array.from({length:130},(_,i)=>`${i%26},${i%26+.25}`)].join('\n'));
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{}),args:['--no-sandbox']});
 try {
  const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.env.ACAN_TEST_URL||'http://127.0.0.1:4000');
  await page.locator('#upload-data input[type=file]').setInputFiles(input);
  await page.locator('#uplColId').filter({hasText:'Target suggerito'}).waitFor();
  await page.getByText('Model',{exact:true}).click();
  await page.locator('#button-learn').filter({hasText:'Apprendi a prevedere'}).waitFor();
  await page.locator('#button-learn').click();
  await page.waitForFunction(()=>/Completato|❌/.test(document.querySelector('#learning-status').innerText),{},{timeout:90000});
  const status=await page.locator('#learning-status').innerText();assert.match(status,/Completato/);
  const download=page.waitForEvent('download');await page.locator('#btn-download').click();
  const saved=await(await download).path();const graph=JSON.parse(await fs.readFile(saved,'utf8'));
  const root=graph.clusterModel[graph.rootId];
  assert.equal(root.cols.filter(c=>c.startsWith('class_')).length,26);
  assert.equal(root.routing_parameters.training_rows,104);
  assert.deepEqual(errors,[]);
  await page.getByText('Generative',{exact:true}).click();
  await page.locator('#lm-history-mode').waitFor();
  assert.equal(await page.locator('#lm-history-prev').count(),1);
  console.log('26 numeric decimal classes: default training and saved schema verified; history navigation visible.');
 } finally {await browser.close();await fs.rm(temporary,{recursive:true,force:true});}
})().catch(e=>{console.error(e);process.exit(1)});
