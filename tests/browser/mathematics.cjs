const {chromium}=require('playwright');
const assert=require('node:assert/strict');
(async()=>{
  const browser=await chromium.launch({headless:true,
    ...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{}),args:['--no-sandbox']});
  try {
    for(const width of [550,1440]) {
      const page=await browser.newPage({viewport:{width,height:1000}});
      await page.goto(process.env.ACAN_TEST_URL||'http://127.0.0.1:4000',{waitUntil:'domcontentloaded'});
      await page.getByText('Matematica',{exact:true}).first().click();
      await page.getByRole('heading',{name:'Passo 5 · Predire la colonna che manca'}).waitFor();
      await page.waitForFunction(()=>document.querySelectorAll('#page-mathematics mjx-container').length>=5);
      assert.equal(await page.locator('#page-mathematics [data-mjx-error]').count(),0);
      await page.getByText('Informatica',{exact:true}).first().click();
      await page.getByRole('heading',{name:'Un testo: dalla coda al checkpoint'}).waitFor();
      assert(await page.getByRole('heading',{name:'Una domanda: dal prompt al prossimo token'}).isVisible());
      console.log('Verified the shared guide at width',width);
      await page.close();
    }
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
