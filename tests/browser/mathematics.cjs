// Mathematical explanations must remain complete and render without TeX errors.
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
      await page.getByRole('heading',{name:'7. Cosa dimostrano le verifiche'}).waitFor();
      await page.waitForFunction(()=>document.querySelectorAll('#page-mathematics mjx-container').length>=15);
      assert.equal(await page.locator('#page-mathematics [data-mjx-error]').count(),0);
      assert.equal(await page.locator('#page-mathematics h3').count(),7);
      const text=await page.locator('#page-mathematics').innerText();
      assert(text.includes('cp-joint-v1')&&text.includes('95,78%'));
      console.log(`Mathematics: ${width}px, all sections and formulas rendered without errors.`);
      await page.close();
    }
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
