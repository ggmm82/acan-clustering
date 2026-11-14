"""Text out of the files a person has: Markdown, Python, ZIP, EPUB, DOCX and PDF.

Every file becomes JSONL records with a ``text`` field (and ``source``, the file it came from),
the format the training already reads. Long texts are cut at paragraph boundaries into pieces of
at most a million characters, because one record is read whole. Nothing is executed: Python files
are text like any other, and archives are read without unpacking them to disk.
"""
import html
from html.parser import HTMLParser
import io
import json
from pathlib import PurePosixPath
import re
import zipfile

TEXT = ('.txt', '.md', '.markdown', '.py', '.rst', '.csv', '.json', '.html', '.htm', '.xhtml', '.tex')
ACCEPTED = ('.txt', '.jsonl') + ('.md', '.markdown', '.py', '.zip', '.epub', '.docx', '.pdf')
PIECE = 1_000_000
MAX_MEMBERS = 100_000
MAX_EXPANDED = 8*1024**3


class _Text(HTMLParser):
    """Visible text of an HTML or XHTML page, paragraphs kept."""
    BLOCKS = {'p', 'div', 'br', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'tr', 'section', 'article', 'blockquote', 'pre'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'head'):
            self.skip += 1
        elif tag in self.BLOCKS:
            self.parts.append('\n\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'head') and self.skip:
            self.skip -= 1
        elif tag in self.BLOCKS:
            self.parts.append('\n\n')

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def text(self):
        return re.sub(r'\n\s*\n+', '\n\n', re.sub(r'[ \t]+', ' ', ''.join(self.parts))).strip()


def html_text(raw):
    parser = _Text()
    parser.feed(raw)
    return parser.text()


def docx_text(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        xml = archive.read('word/document.xml').decode('utf-8', errors='replace')
    paragraphs = []
    for paragraph in re.findall(r'<w:p[ >].*?</w:p>', xml, flags=re.S):
        words = re.findall(r'<w:t(?: [^>]*)?>(.*?)</w:t>', paragraph, flags=re.S)
        if words:
            paragraphs.append(html.unescape(''.join(words)))
    return '\n\n'.join(paragraphs)


def epub_text(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(('.xhtml', '.html', '.htm'))]
        order = _spine(archive) or sorted(names)
        return '\n\n'.join(html_text(archive.read(name).decode('utf-8', errors='replace')) for name in order if name in names)


def _spine(archive):
    """The reading order of an EPUB, from its package file."""
    try:
        container = archive.read('META-INF/container.xml').decode('utf-8', errors='replace')
        package = re.search(r'full-path="([^"]+)"', container).group(1)
        content = archive.read(package).decode('utf-8', errors='replace')
    except (KeyError, AttributeError):
        return None
    base = str(PurePosixPath(package).parent)
    items = dict(re.findall(r'<item [^>]*id="([^"]+)"[^>]*href="([^"]+)"', content))
    items.update({key: value for value, key in re.findall(r'<item [^>]*href="([^"]+)"[^>]*id="([^"]+)"', content)})
    order = []
    for reference in re.findall(r'<itemref [^>]*idref="([^"]+)"', content):
        if reference in items:
            path = items[reference] if base in ('', '.') else f'{base}/{items[reference]}'
            order.append(str(PurePosixPath(path)))
    return order


def pdf_text(raw):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ValueError('Per leggere i PDF il servizio deve avere pypdf (aggiorna il worker).') from None
    reader = PdfReader(io.BytesIO(raw))
    return '\n\n'.join((page.extract_text() or '').strip() for page in reader.pages)


def pieces(text):
    """Cut at paragraph boundaries into pieces of at most ``PIECE`` characters."""
    text = text.strip()
    while len(text) > PIECE:
        cut = text.rfind('\n\n', 0, PIECE)
        cut = cut if cut > PIECE//2 else PIECE
        yield text[:cut].strip()
        text = text[cut:].strip()
    if text:
        yield text


def texts(name, raw, depth=0):
    """(source, text) for every readable document inside the file ``name``."""
    suffix = PurePosixPath(name.lower()).suffix
    if suffix in ('.html', '.htm', '.xhtml'):
        yield name, html_text(raw.decode('utf-8', errors='replace'))
    elif suffix in TEXT:
        yield name, raw.decode('utf-8', errors='replace')
    elif suffix == '.jsonl':
        for line in raw.decode('utf-8', errors='replace').splitlines():
            record = json.loads(line) if line.strip() else None
            if isinstance(record, dict) and isinstance(record.get('text'), str):
                yield name, record['text']
    elif suffix == '.docx':
        yield name, docx_text(raw)
    elif suffix == '.epub':
        yield name, epub_text(raw)
    elif suffix == '.pdf':
        yield name, pdf_text(raw)
    elif suffix == '.zip' and depth < 2:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) > MAX_MEMBERS or sum(item.file_size for item in members) > MAX_EXPANDED:
                raise ValueError('Archivio troppo grande una volta espanso (massimo 100.000 file e 8 GB).')
            for item in members:
                if '__MACOSX' in item.filename or PurePosixPath(item.filename).name.startswith('.'):
                    continue
                yield from texts(item.filename, archive.read(item), depth+1)


def convert(path, name):
    """Rewrite the uploaded file at ``path`` as JSONL; returns the number of records."""
    with open(path, 'rb') as stream:
        raw = stream.read()
    count = 0
    with open(path, 'w', encoding='utf-8') as output:
        for source, text in texts(name, raw):
            for piece in pieces(text):
                output.write(json.dumps(dict(text=piece, source=source), ensure_ascii=False)+'\n')
                count += 1
    if not count:
        raise ValueError('Nessun testo leggibile nel file (un PDF fatto di sole immagini non contiene testo).')
    return count
