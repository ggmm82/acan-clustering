// The step-by-step mathematics must stay complete and render without TeX errors.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
(async()=>{
  const browser=await chromium.launch({headless:true,
    ...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{}),args:['--no-sandbox']});
  try {
    for(const width of [550,1440]) {
      const page=await browser.newPage({viewport:{width,height:1000}});
      await page.goto(process.env.ACAN_TEST_URL||'http://127.0.0.1:4000',{waitUntil:'domcontentloaded'});
      await page.getByText('Matematica',{exact:true}).click();
      await page.getByRole('heading',{name:'Che cosa garantiscono le verifiche, e i limiti'}).waitFor();
      await page.waitForFunction(()=>document.querySelectorAll('#page-mathematics mjx-container').length>=15);
      assert.equal(await page.locator('#page-mathematics [data-mjx-error]').count(),0);
      // One story of one model: ten steps for table rows and text positions together, the graph, the guarantees.
      assert.equal(await page.locator('#page-mathematics h3').count(),12);
      for(const name of ['Il modello, passo per passo','Passo 1 · I dati diventano righe','Passo 9 · Valori che si somigliano: i concetti dei token e i percorsi'])
        assert(await page.getByRole('heading',{name}).isVisible());
      const text=await page.locator('#page-mathematics').innerText();
      assert(text.includes('Passo 6 · Contare a blocchi')&&text.includes('Passo 7 · Generalizzare dove chi legge perde meno'));
      assert(!text.includes('Parte A')&&!text.includes('Parte B'),'one story, no separate parts');
      assert(text.includes('Il testo è una tabella'));
      assert(text.includes('Senza target')&&text.includes('Una tabella più grande della memoria'));
      for(const gone of ['cp-v9','SmolLM2','acan/algebra.py','revisione precedente'])
        assert(!text.includes(gone),gone);
      console.log(`Mathematics: ${width}px, all sections and formulas rendered without errors.`);
      await page.close();
    }
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
