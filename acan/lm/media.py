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
MARKERS = ['<|image|>', '<|/image|>', '<|row|>', '<|audio|>', '<|/audio|>', '<|video|>', '<|/video|>', '<|frame|>']
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


def encode_audio(raw):
    """A WAV file: mono, resampled to 8 kHz, 8-bit mu-law."""
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
    if rate != AUDIO_RATE and len(samples):
        times = np.arange(int(len(samples)*AUDIO_RATE/rate))/AUDIO_RATE
        samples = np.interp(times, np.arange(len(samples))/rate, samples)
    mu = 255.
    coded = np.sign(samples)*np.log1p(mu*np.abs(np.clip(samples, -1, 1)))/np.log1p(mu)
    codes = np.clip(np.round((coded+1)/2*255), 0, 255).astype(int)
    return '<|audio|>'+''.join(f'<|a{code}|>' for code in codes)+'<|/audio|>'


def encode(name, raw):
    suffix = name.lower().rsplit('.', 1)[-1]
    if '.'+suffix in IMAGE_EXTENSIONS:
        return encode_image(raw)
    if '.'+suffix in AUDIO_EXTENSIONS:
        return encode_audio(raw)
    if '.'+suffix in VIDEO_EXTENSIONS:
        return encode_video(raw)
    raise ValueError('Formato multimediale non supportato: '+name)


# ---- back from tokens ----------------------------------------------------------------------------------------

def _decode_grid(body, size):
    from PIL import Image
    image = Image.new('RGB', (size, size))
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


def decode_audio(body):
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
        stream.setframerate(AUDIO_RATE)
        stream.writeframes((np.clip(samples, -1, 1)*32767).astype('<i2').tobytes())
    return buffer.getvalue()


PATTERN = re.compile(r'<\|(image|audio|video)\|>(.*?)(?:<\|/\1\|>|$)', re.S)


def parts(text):
    """The text split into ('text', str) and ('image'|'audio'|'video', bytes) pieces, for display."""
    found, start = [], 0
    for match in PATTERN.finditer(text):
        if match.start() > start:
            found.append(('text', text[start:match.start()]))
        kind, body = match.group(1), match.group(2)
        data = {'image': decode_image, 'audio': decode_audio, 'video': decode_video}[kind](body)
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
