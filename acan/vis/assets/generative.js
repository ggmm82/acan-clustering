/* Stream corpus chunks; never base64 an entire multi-GB training file in Dash. */
(function () {
  async function call(action, body, headers) {
    const response = await fetch('/generative-upload/' + action, {
      method: 'POST', body: body,
      headers: Object.assign({'X-ACAN-Workspace': window.acanLmWorkspace || ''}, headers || {})
    });
    const result = await response.json();
    if (!response.ok) throw Error(result.error || 'Caricamento fallito');
    return result;
  }
  function bind() {
    const zone = document.getElementById('lm-file-upload');
    if (zone && !zone.dataset.bound) {
      zone.dataset.bound = 'true';
      const input = document.createElement('input'); input.type='file'; input.accept='.txt,.jsonl,.md,.markdown,.py,.zip,.epub,.docx,.pdf';
      input.setAttribute('aria-label','Corpus di training TXT o JSONL');
      const button = document.createElement('button'); button.textContent='Carica il tuo testo'; button.className='btn btn-outline-primary my-2';
      zone.append(input, button);
      button.onclick = async function () {
        const status = document.getElementById('lm-upload-status');
        try {
          if (!input.files.length || !window.acanLmWorkspace) throw Error('Seleziona un corpus e attendi il workspace.');
          button.disabled = true;
          const file = input.files[0];
          const corpus = await call('begin',JSON.stringify({name:file.name,size:file.size}),{'Content-Type':'application/json'});
          const size = 1024*1024;
          for(let offset=0;offset<file.size;offset+=size) {
            await call('chunk',file.slice(offset,offset+size),{'X-Corpus-ID':corpus.id,'X-Offset':String(offset),'Content-Type':'application/octet-stream'});
            status.textContent='Caricamento '+Math.min(100,Math.round((offset+size)/file.size*100))+'%';
          }
          const ready = await call('finish',null,{'X-Corpus-ID':corpus.id});
          status.textContent='Corpus salvato sul volume persistente: '+file.name;
          window.dash_clientside.set_props('lm-upload-state',{data:ready});
        } catch(error) { status.textContent=error.message; }
        finally { button.disabled=false; }
      };
    }
    const recovery = document.getElementById('lm-recovery-input');
    if (recovery && !recovery.dataset.bound) {
      recovery.dataset.bound='true';
      const label=document.createElement('label');label.textContent='Ritrova i progetti con una chiave di recupero: ';
      const input=document.createElement('input');input.type='file';input.accept='.json';
      input.onchange=async()=>{
        try {
          const data=JSON.parse(await input.files[0].text());
          if(!/^[0-9a-f]{64}$/.test(data.acan_lm_workspace))throw Error('Chiave non valida');
          window.acanLmWorkspace=data.acan_lm_workspace;
          window.dash_clientside.set_props('lm-workspace',{data:data.acan_lm_workspace});
        } catch(error){window.alert(error.message);}
      };
      label.append(input);recovery.append(label);
    }
  }
  new MutationObserver(bind).observe(document.documentElement,{childList:true,subtree:true});
  window.addEventListener('load',bind);
})();
