"""Images, sounds and short videos as tokens the language model can read and write.

The model reads one sequence of tokens. A medium becomes a run of special tokens between markers,
written as text, so the same corpus, tokenizer and training read it:

* image: ``<|image|>`` then 32 rows of 32 pixels ``<|pN|>`` each followed by ``<|row|>``, then
  ``<|/image|>``. N is one of 256 colours (3 bits red, 3 green, 2 blue);
* sound: ``<|audio|>`` then 8,000 samples per second ``<|aN|>`` (8-bit mu-law, mono), then
  ``<|/audio|>``; at most ``AUDIO_SECONDS`` seconds;
* video: ``<|video|>`` then frames of 16 x 16 pixels, each ``<|frame|>`` rows ``<|row|>``, at most
  ``VIDEO_FRAMES`` frames at ``VIDEO_FPS`` per second, then ``<|/video|>``.

These are fixed, simple codes, readable back without a learned network: enough to learn and to
look at small pictures, simple sounds and short animations. Real photographs, speech and music in
quality need learned codecs (a neural tokenizer), trained on a GPU.

WAV, GIF and pictures are read with the libraries of the worker. Compressed sound (MP3, OGG, FLAC,
M4A, AAC, Opus) and video (MP4, WebM, MOV, AVI, MKV) need ffmpeg, which is not installed with the
worker: the first such file downloads a pinned, checksum-verified static build and keeps it.
"""
import io
import re
import wave

IMAGE_SIZE = 32
VIDEO_SIZE = 16
VIDEO_FRAMES = 64
VIDEO_FPS = 4
AUDIO_RATE = 8000
AUDIO_SECONDS = 10
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
AUDIO_EXTENSIONS = ('.wav',)
VIDEO_EXTENSIONS = ('.gif',)
# Read through ffmpeg, which is not part of the worker image: it is downloaded, verified and kept
# the first time one of these files arrives (see ``ffmpeg``).
FFMPEG_AUDIO = ('.mp3', '.ogg', '.flac', '.m4a', '.aac', '.opus')
FFMPEG_VIDEO = ('.mp4', '.webm', '.mov', '.avi', '.mkv')
FFMPEG_WHEELS = {   # imageio-ffmpeg 0.7.0 (BSD-2-Clause wrapper around a static ffmpeg 7.0.2 build), from PyPI
    'x86_64': dict(url='https://files.pythonhosted.org/packages/04/96/15fe7ef562f7494c358cbd9b521d2da82d7a331b5513703756166478bd5f/'
                       'imageio_ffmpeg-0.7.0-py3-none-manylinux2014_x86_64.whl',
                   bytes=29498414, sha256='4e1f36bef7bedd4bc5dc25f60a8345d88f57133fa1e754f09b411ac61b854408'),
    'aarch64': dict(url='https://files.pythonhosted.org/packages/9c/68/5c7eac9733c0a35b46f939df7079bbf72463dc347d4ff51fbad138d15ba9/'
                        'imageio_ffmpeg-0.7.0-py3-none-manylinux2014_aarch64.whl',
                    bytes=25632882, sha256='588a23c6a1c55a7cb51febfe97d3ef8dd5e7fa0ef7e14c7da8828e2886bdd3a3'),
}
MARKERS = ['<|image|>', '<|/image|>', '<|row|>', '<|audio|>', '<|/audio|>', '<|video|>', '<|/video|>', '<|frame|>',
           '<|audio4|>', '<|/audio4|>']      # sound at 4 kHz: half the tokens, so a short word fits in one piece
TOKENS = MARKERS + [f'<|p{n}|>' for n in range(256)] + [f'<|a{n}|>' for n in range(256)]


def _colour(r, g, b):
    return (r >> 5) << 5 | (g >> 5) << 2 | (b >> 6)


def _rgb(code):
    return (((code >> 5) & 7)*255//7, ((code >> 2) & 7)*255//7, (code & 3)*255//3)


def _grid(image, size):
    from PIL import Image
    image = image.convert('RGB')
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (size, size), (0, 0, 0))
    canvas.paste(image, ((size-image.width)//2, (size-image.height)//2))
    pixels = list(canvas.getdata())
    return ''.join(''.join(f'<|p{_colour(*pixels[row*size+column])}|>' for column in range(size))+'<|row|>' for row in range(size))


def encode_image(raw):
    from PIL import Image
    return '<|image|>'+_grid(Image.open(io.BytesIO(raw)), IMAGE_SIZE)+'<|/image|>'


def encode_video(raw):
    """An animated GIF: frames sampled at ``VIDEO_FPS`` per second."""
    from PIL import Image, ImageSequence
    animation = Image.open(io.BytesIO(raw))
    frames, elapsed, next_time = [], 0., 0.
    for frame in ImageSequence.Iterator(animation):
        if elapsed >= next_time:
            frames.append('<|frame|>'+_grid(frame.copy(), VIDEO_SIZE))
            next_time += 1/VIDEO_FPS
            if len(frames) == VIDEO_FRAMES:
                break
        elapsed += (frame.info.get('duration') or 100)/1000
    return '<|video|>'+''.join(frames)+'<|/video|>'


def encode_audio(raw, target=AUDIO_RATE, trim=False):
    """A WAV file: mono, resampled to ``target`` (8 or 4 kHz), 8-bit mu-law; ``trim`` drops the silence around it."""
    import numpy as np
    with wave.open(io.BytesIO(raw)) as stream:
        channels, width, rate = stream.getnchannels(), stream.getsampwidth(), stream.getframerate()
        frames = stream.readframes(min(stream.getnframes(), rate*AUDIO_SECONDS))
    kind = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if kind is None:
        raise ValueError('WAV non supportato: usare campioni a 8, 16 o 32 bit.')
    samples = np.frombuffer(frames, dtype=kind).astype(np.float64)
    samples = (samples-128)/128 if width == 1 else samples/float(2**(8*width-1))
    samples = samples.reshape(-1, channels).mean(1)
    if rate != target and len(samples):
        times = np.arange(int(len(samples)*target/rate))/target
        samples = np.interp(times, np.arange(len(samples))/rate, samples)
    if trim and len(samples):
        loud = np.flatnonzero(np.abs(samples) > .05*np.abs(samples).max())
        samples = samples[loud[0]:loud[-1]+1] if len(loud) else samples
        samples = samples[:1990]                                # with the markers and the request, within 2,048 tokens
    mu = 255.
    coded = np.sign(samples)*np.log1p(mu*np.abs(np.clip(samples, -1, 1)))/np.log1p(mu)
    codes = np.clip(np.round((coded+1)/2*255), 0, 255).astype(int)
    marker = 'audio' if target == AUDIO_RATE else 'audio4'
    return f'<|{marker}|>'+''.join(f'<|a{code}|>' for code in codes)+f'<|/{marker}|>'


def ffmpeg(tools):
    """Path of an ffmpeg program: the system one if present, else downloaded once into ``tools`` and verified."""
    import fcntl
    import hashlib
    import os
    import platform
    import shutil
    import zipfile
    from pathlib import Path
    found = shutil.which('ffmpeg')
    if found:
        return found
    if tools is None:
        raise ValueError('Per questo formato serve ffmpeg: caricalo dal laboratorio, che lo scarica alla prima occorrenza.')
    machine = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(platform.machine().lower(), platform.machine().lower())
    wheel = FFMPEG_WHEELS.get(machine)
    if wheel is None:
        raise ValueError(f'ffmpeg non disponibile per questa macchina ({machine}): convertire in WAV o GIF.')
    tools = Path(tools)
    tools.mkdir(parents=True, exist_ok=True)
    target = tools/f"ffmpeg-{wheel['sha256'][:12]}"
    with (tools/'ffmpeg.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)                         # two uploads at once download it once
        if target.exists():
            return str(target)
        from .dialogue_data import download
        package = download(dict(url=wheel['url'], path='imageio-ffmpeg.whl', bytes=wheel['bytes'], sha256=wheel['sha256']),
                           tools/'imageio-ffmpeg.whl')
        with zipfile.ZipFile(package) as archive:
            member = next(name for name in archive.namelist() if name.startswith('imageio_ffmpeg/binaries/ffmpeg-'))
            temporary = target.with_suffix('.tmp')
            temporary.write_bytes(archive.read(member))
        os.chmod(temporary, 0o755)
        os.replace(temporary, target)
        package.unlink()
        hashlib.sha256(target.read_bytes())                     # readable whole: the copy is complete
    return str(target)


def _run(tools, arguments, raw):
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.media') as source:
        source.write(raw)
        source.flush()
        result = subprocess.run([ffmpeg(tools), '-nostdin', '-loglevel', 'error', '-i', source.name, *arguments, '-'],
                                capture_output=True, timeout=600)
    if result.returncode != 0:
        raise ValueError('File multimediale illeggibile: '+result.stderr.decode(errors='replace')[-300:])
    return result.stdout


def encode_compressed_audio(raw, tools):
    """MP3, OGG, FLAC, M4A, AAC, Opus through ffmpeg: mono, 8 kHz, then the same mu-law codes."""
    import numpy as np
    pcm = _run(tools, ['-t', str(AUDIO_SECONDS), '-ac', '1', '-ar', str(AUDIO_RATE), '-f', 's16le'], raw)
    samples = np.frombuffer(pcm, dtype='<i2').astype(np.float64)/32768
    mu = 255.
    coded = np.sign(samples)*np.log1p(mu*np.abs(samples))/np.log1p(mu)
    codes = np.clip(np.round((coded+1)/2*255), 0, 255).astype(int)
    return '<|audio|>'+''.join(f'<|a{code}|>' for code in codes)+'<|/audio|>'


def encode_compressed_video(raw, tools):
    """MP4, WebM, MOV, AVI, MKV through ffmpeg: frames of 16 x 16 at the same rate as GIF animations."""
    size = VIDEO_SIZE
    scale = f'fps={VIDEO_FPS},scale={size}:{size}:force_original_aspect_ratio=decrease,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2'
    pixels = _run(tools, ['-vf', scale, '-frames:v', str(VIDEO_FRAMES), '-f', 'rawvideo', '-pix_fmt', 'rgb24'], raw)
    frame = size*size*3
    frames = []
    for start in range(0, len(pixels)-frame+1, frame):
        data = pixels[start:start+frame]
        rows = [''.join(f'<|p{_colour(*data[(row*size+column)*3:(row*size+column)*3+3])}|>' for column in range(size))+'<|row|>'
                for row in range(size)]
        frames.append('<|frame|>'+''.join(rows))
    if not frames:
        raise ValueError('Video senza fotogrammi leggibili.')
    return '<|video|>'+''.join(frames)+'<|/video|>'


def encode(name, raw, tools=None):
    suffix = '.'+name.lower().rsplit('.', 1)[-1]
    if suffix in IMAGE_EXTENSIONS:
        return encode_image(raw)
    if suffix in AUDIO_EXTENSIONS:
        return encode_audio(raw)
    if suffix in VIDEO_EXTENSIONS:
        return encode_video(raw)
    if suffix in FFMPEG_AUDIO:
        return encode_compressed_audio(raw, tools)
    if suffix in FFMPEG_VIDEO:
        return encode_compressed_video(raw, tools)
    raise ValueError('Formato multimediale non supportato: '+name)


# ---- back from tokens ----------------------------------------------------------------------------------------

def _decode_grid(body, size):
    from PIL import Image
    image = Image.new('RGB', (size, size), (128, 128, 128))      # rows not written stay grey, not black: never mistaken for content
    rows = [row for row in body.split('<|row|>')][:size]
    for y, row in enumerate(rows):
        for x, code in enumerate(re.findall(r'<\|p(\d+)\|>', row)[:size]):
            image.putpixel((x, y), _rgb(int(code)))
    return image


def decode_image(body):
    buffer = io.BytesIO()
    _decode_grid(body, IMAGE_SIZE).resize((IMAGE_SIZE*8, IMAGE_SIZE*8), 0).save(buffer, 'PNG')
    return buffer.getvalue()


def decode_video(body):
    frames = [_decode_grid(part, VIDEO_SIZE).resize((VIDEO_SIZE*12, VIDEO_SIZE*12), 0) for part in body.split('<|frame|>') if part.strip()]
    if not frames:
        return None
    buffer = io.BytesIO()
    frames[0].save(buffer, 'GIF', save_all=True, append_images=frames[1:], duration=int(1000/VIDEO_FPS), loop=0)
    return buffer.getvalue()


def decode_audio(body, rate=AUDIO_RATE):
    import numpy as np
    codes = np.array([int(code) for code in re.findall(r'<\|a(\d+)\|>', body)], dtype=np.float64)
    if not len(codes):
        return None
    mu = 255.
    coded = codes/255*2-1
    samples = np.sign(coded)*((1+mu)**np.abs(coded)-1)/mu
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((np.clip(samples, -1, 1)*32767).astype('<i2').tobytes())
    return buffer.getvalue()


PATTERN = re.compile(r'<\|(image|audio4|audio|video)\|>(.*?)(?:<\|/\1\|>|$)', re.S)


def parts(text):
    """The text split into ('text', str) and ('image'|'audio'|'video', bytes) pieces, for display."""
    found, start = [], 0
    for match in PATTERN.finditer(text):
        if match.start() > start:
            found.append(('text', text[start:match.start()]))
        kind, body = match.group(1), match.group(2)
        data = {'image': decode_image, 'audio': decode_audio, 'audio4': lambda part: decode_audio(part, AUDIO_RATE//2),
                'video': decode_video}[kind](body)
        kind = 'audio' if kind == 'audio4' else kind
        if data:
            found.append((kind, data))
        start = match.end()
    if start < len(text):
        found.append(('text', text[start:]))
    return found


def readable(tokenizer, ids):
    """Decoded text that keeps media tokens (so they can be shown) and drops the conversation markers."""
    if tokenizer.convert_tokens_to_ids('<|image|>') in (None, tokenizer.unk_token_id):
        return tokenizer.decode(ids, skip_special_tokens=True)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    for marker in ('<|endoftext|>', '<|im_start|>', '<|im_end|>'):
        text = text.replace(marker, '')
    return text
