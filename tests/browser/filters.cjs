// No user data or external app access: generate a synthetic Titanic-shaped table.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');

(async () => {
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'acan-filters-'));
  const csv = path.join(temporary, 'synthetic.csv');
  const rows = ['PassengerId,Name,Ticket,Cabin,Age,Survived'];
  for (let i = 0; i < 120; i++) rows.push(`${i},Person${i},Ticket${i},Cabin${i},${18+i%60},${i%2}`);
  await fs.writeFile(csv, rows.join('\n'));
  const browser = await chromium.launch({headless: true,
    ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}), args: ['--no-sandbox']});
  const errors = [];
  try {
    for (const width of [550, 1440]) {
      const context = await browser.newContext({viewport: {width, height: 1100}, acceptDownloads: true});
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(process.env.ACAN_TEST_URL || 'http://127.0.0.1:4000', {waitUntil: 'domcontentloaded'});
      await page.locator('#upload-data input[type=file]').setInputFiles(csv);
      await page.locator('#uplColId').filter({hasText: 'Target suggerito'}).waitFor();
      await page.getByText('Modello', {exact:true}).click();

      async function learn() {
        const sent = page.waitForRequest(request => request.method() === 'POST' &&
          request.url().includes('_dash-update-component') &&
          request.postDataJSON()?.inputs?.some(input => input.id === 'button-learn'));
        const response = page.waitForResponse(response => response.request().method() === 'POST' &&
          response.url().includes('_dash-update-component') &&
          response.request().postDataJSON()?.inputs?.some(input => input.id === 'button-learn'));
        await page.locator('#button-learn').click();
        const [request, reply] = await Promise.all([sent, response]);
        const payload = request.postDataJSON();
        const filters = payload.state.find(item => item.id === 'dpdnFilter').value;
        assert.deepEqual(payload.state.find(item => item.id === 'label-store').value, ['Survived']);
        console.log(JSON.stringify({width, filters}));
        assert.deepEqual([...filters].sort(), ['Cabin','Name','PassengerId','Ticket']);
        const body = await reply.json();
        assert.equal(body.response['learning-monitor'].disabled, false, JSON.stringify(body));
        await page.waitForFunction(() => /Completato|❌/.test(document.querySelector('#learning-status').innerText), {}, {timeout:60000});
        assert.match(await page.locator('#learning-status').innerText(), /Completato/);
      }
      await learn(); // Immediate click waits for synchronized target/filter controls.
      await page.getByText('Impostazioni avanzate', {exact:true}).first().click();
      const downloadEvent = page.waitForEvent('download');
      await page.locator('#btn-download').click();
      const model = path.join(temporary, `model-${width}.json`);
      await (await downloadEvent).saveAs(model);
      await page.locator('#upload-model input[type=file]').setInputFiles(model);
      await page.locator('#learning-status').filter({hasText:'Modello caricato'}).waitFor();
      await learn(); // Loading model targets must not erase feature exclusions.
      await page.locator('#btn-clear-filter').click();
      await page.locator('#btn-suggest-filter').click();
      await learn();
      await context.close();
      console.log(`Browser filters: ${width}px upload, model reload and explicit restore passed.`);
    }
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    await fs.rm(temporary, {recursive:true, force:true});
  }
})().catch(error => {console.error(error);process.exitCode=1;});
