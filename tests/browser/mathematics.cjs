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
      await page.getByRole('heading',{name:'Parte D · Che cosa è stato misurato'}).waitFor();
      await page.waitForFunction(()=>document.querySelectorAll('#page-mathematics mjx-container').length>=15);
      assert.equal(await page.locator('#page-mathematics [data-mjx-error]').count(),0);
      // Step by step: a table (9 steps and two notes), then a text as a table of the same model (10 steps).
      assert.equal(await page.locator('#page-mathematics h3').count(),21);
      for(const name of ['Parte A · Una tabella','Parte B · Un testo','Parte C · Tabella e testo a confronto'])
        assert(await page.getByRole('heading',{name}).isVisible());
      const text=await page.locator('#page-mathematics').innerText();
      assert(text.includes('Passo 4 · Una passata a blocchi')&&text.includes('Passo 8 · Imparare è contare, a blocchi')&&text.includes('Passo 6 · I percorsi verso il token'));
      assert(text.includes('Il testo è una tabella'));
      assert(text.includes('Senza target')&&text.includes('Una tabella più grande della memoria'));
      for(const gone of ['cp-v9','SmolLM2','acan/algebra.py','revisione precedente'])
        assert(!text.includes(gone),gone);
      console.log(`Mathematics: ${width}px, all sections and formulas rendered without errors.`);
      await page.close();
    }
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
