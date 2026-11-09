#!/usr/bin/env python3
"""make_infographics.py — regenerate README/docs SVG infographics from the
data embedded in the current assets, using a guaranteed-legible layout:
minimum font sizes, PIL-measured text wrapping, auto-stacked lines per card.
Usage: python3 Scripts/make_infographics.py [assets-dir]"""
import re, sys, glob, html, os
from PIL import ImageFont

FONT_R = "/System/Library/Fonts/Helvetica.ttc"
FONT_B = "/System/Library/Fonts/HelveticaNeue.ttc"
_r = ImageFont.truetype(FONT_R, 100)
_b = ImageFont.truetype(FONT_B, 100)
def tw(s, fs, bold=False):  # text width in px at font-size fs
    return ( _b if bold else _r ).getlength(s) * fs / 100.0

BIG_RE = re.compile(r'^[0-9][0-9.,]*\s*(x|×|%|GiB|tok/s|K|B|M|s|ms)?$')
CAP_RE = re.compile(r'^[A-Z0-9 ·&+/()\-–—.:,%×x<>=\[\]]+$')

def parse(path_or_text):
    s = path_or_text if '<svg' in path_or_text[:200] else open(path_or_text).read()
    vb = [float(v) for v in re.search(r'viewBox="([\d. ]+)"', s).group(1).split()]
    rects, texts = [], []
    in_defs = False
    stack = [(0.0, 0.0)]
    css = {c: float(f) for c, f in re.findall(r'\.([\w-]+)\{[^}]*?font-size:([\d.]+)px', s)}  # cumulative translate offsets
    # walk every open/close tag in order; group translates nest
    for m in re.finditer(r'(<defs>|</defs>|<g\b[^>]*>|</g>|<rect\b[^>]*?/?>|<text\b[^>]*>.*?</text>)', s, re.S):
        tok = m.group(1)
        if tok == '<defs>': in_defs = True; continue
        if tok == '</defs>': in_defs = False; continue
        if in_defs: continue
        if tok.startswith('</g'):
            if len(stack) > 1: stack.pop()
            continue
        if tok.startswith('<g'):
            dx, dy = stack[-1]
            t = re.search(r'transform="translate\(([-\d.]+)[, ]+([-\d.]+)\)"', tok)
            if t: dx, dy = dx + float(t.group(1)), dy + float(t.group(2))
            stack.append((dx, dy))
            continue
        ox, oy = stack[-1]
        if tok.startswith('<rect'):
            a = tok
            g = lambda k: re.search(rf'{k}="([-\d.]+)"', a)
            if g('width') and g('height'):
                r = dict(x=float(g('x').group(1)) if g('x') else 0.0, y=float(g('y').group(1)) if g('y') else 0.0, y0=0,
                         w=float(g('width').group(1)), h=float(g('height').group(1)))
                r['y'] = r['y'] + oy; r['x'] = r['x'] + ox; r.pop('y0')
                f = re.search(r'stroke="([^"]+)"', a)
                r['stroke'] = f.group(1) if f and f.group(1) != 'none' else None
                if r['w'] >= 180 and r['h'] >= 90 and r['w'] < vb[2] - 2: rects.append(r)
            continue
        a = re.match(r'<text([^>]*)>', tok).group(1)
        t = html.unescape(re.sub(r'<[^>]+>', '', tok[len('<text'+a)+1:])).strip()
        if not t: continue
        x, y = re.search(r'x="([-\d.]+)"', a), re.search(r'y="([-\d.]+)"', a)
        if not (x and y): continue
        fs = re.search(r'font-size="([-\d.]+)"', a)
        if fs: fsz = float(fs.group(1))
        else:
            cls = re.search(r'class="([^"]+)"', a)
            sizes = [css[c] for c in (cls.group(1).split() if cls else []) if c in css]
            fsz = max(sizes) if sizes else 12.0
        fill = re.search(r'fill="([^"]+)"', a)
        anchor = 'middle' if 'text-anchor="middle"' in a else ('end' if 'text-anchor="end"' in a else 'start')
        texts.append(dict(x=float(x.group(1)) + ox, y=float(y.group(1)) + oy, rot='rotate(' in a,
                          fs=fsz,
                          fill=fill.group(1) if fill else '#cbd5e1', anchor=anchor, t=t))
    return vb, rects, texts

def assign(vb, rects, texts):
    """cluster texts into header, panels, and a leftovers card (nothing dropped)."""
    LOGO = {'Q', 'QWEGN', 'Qwegn', 'INFERENCE ENGINE'}
    texts = [t for t in texts if t['t'].strip() not in LOGO]
    top = min((r['y'] for r in rects), default=vb[3])
    header = [t for t in texts if t['y'] < top - 4 and t['y'] < vb[3] - 26]
    used = {id(t) for t in header}
    rest = [t for t in texts if id(t) not in used and t['y'] < vb[3] - 26]
    footer = [t for t in texts if t['y'] >= vb[3] - 26]
    panels = []
    for r in sorted(rects, key=lambda r: (r['y'], r['x'])):
        grp = [t for t in rest if r['x'] - 2 <= t['x'] <= r['x'] + r['w'] + 2 and r['y'] - 6 <= t['y'] <= r['y'] + r['h']]
        if grp:
            panels.append(dict(rect=r, texts=sorted(grp, key=lambda t: (round(t['y']/8), t['x']))))
            for t in grp: rest.remove(t)
        elif r['h'] >= 150 and r['w'] >= 300:
            panels.append(dict(rect=r, texts=[]))
    if rest:  # keep every datum: bundle unplaced text into a synthetic wide card
        panels.append(dict(rect=dict(x=0, y=vb[3]-30, w=vb[2], h=30, stroke='#38bdf8'),
                           texts=sorted(rest, key=lambda t: (t['y'], t['x']))))
    return header, panels, [], footer

def wrap(t, fs, maxw, bold=False):
    """word-wrap with forced split of over-long single tokens (paths, IDs)."""
    lines, cur = [], ''
    for w in t.split(' '):
        pieces = []
        while w and tw(w, fs, bold) > maxw:
            k = max(1, int(len(w) * maxw / tw(w, fs, bold)))
            pieces.append(w[:k]); w = w[k:]
        if w: pieces.append(w)
        for pc in pieces:
            trial = (cur + ' ' + pc).strip() if cur else pc
            if tw(trial, fs, bold) <= maxw:
                cur = trial
            else:
                if cur: lines.append(cur)
                cur = pc
    if cur: lines.append(cur)
    return [l for l in lines if l.strip()]

def render(name, vb, header, panels, leftover, footer, out):
    ACC = ['#38bdf8', '#c084fc', '#fbbf24', '#34d399', '#f87171', '#818cf8']
    W = 1000
    pad, gap = 24, 20
    n = max(len(panels), 1)
    cols = 3 if n > 3 else (2 if n > 1 else 1)
    rows = (n + cols - 1) // cols
    cw = (W - pad * (cols + 1)) // cols
    # build card content first, then compute row heights from content
    cards = []
    for p in panels:
        ts = p['texts']
        title = ts[0]['t'] if ts and CAP_RE.match(ts[0]['t']) and len(ts[0]['t']) < 46 else (name.replace('_', ' ').title())
        idx = 1 if ts and title == ts[0]['t'] else 0
        big = None
        for t in ts[idx:idx+3]:
            if BIG_RE.match(t['t']) and not t['t'].endswith('tok/s') is False and len(t['t']) <= 9:
                big = t['t']; idx = ts.index(t) + 1; break
        lines = []
        for t in ts[idx:]:
            if big and t['t'] in big: continue
            lines.append(t['t'])
        cards.append((title, big, lines, p['rect']['stroke'] or ACC[len(cards) % len(ACC)]))
    # measure heights
    body_fs, title_fs, big_fs, lead = 16.5, 18, 38, 24
    card_html = []
    row_h = []
    for ci, (title, big, lines, acc) in enumerate(cards):
        y = title_fs + 14
        if big: y += big_fs + 16
        wrapped = []
        for ln in lines:
            for seg in wrap(ln, body_fs, cw - 34):
                wrapped.append(seg)
        y += lead * len(wrapped) + 10
        ch = max(y, 190)
        row_h.append(ch)
        card_html.append((title, big, wrapped, acc))
    ch_per_row = [max(row_h[r*cols:(r+1)*cols] or [0]) for r in range(rows)]
    foot_h = 52
    esc = lambda s: s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    # pre-compute header lines (to know the height before emitting the root tag)
    head_lines = []
    if header:
        hy = 46
        for seg in wrap(header[0]['t'], 26, W - pad*2, True):
            head_lines.append((hy, 26, '#f1f5f9', True, seg)); hy += 32
        sub = "  ·  ".join(t['t'] for t in header[1:]) if len(header) > 1 else ""
        for seg in wrap(sub, 16, W - pad*2)[:4]:
            head_lines.append((hy, 16, '#7dd3fc', False, seg)); hy += 21
        head_h = max(92, hy + 14)
    else:
        head_lines.append((46, 26, '#f1f5f9', True, name.replace('_', ' '))); head_h = 92
    H = int(head_h + sum(ch_per_row) + pad * (rows + 1) + foot_h)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         '<defs><linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#0b1120"/><stop offset="100%" stop-color="#04060a"/></linearGradient>',
         '<linearGradient id="card" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#151d31" stop-opacity="0.97"/><stop offset="100%" stop-color="#0c1122" stop-opacity="0.95"/></linearGradient></defs>',
         f'<rect width="{W}" height="{H}" fill="url(#bg)"/>']
    for hy, fsz, fill, bold, seg in head_lines:
        fw = ' font-weight="bold"' if bold else ''
        fam = 'Helvetica Neue,Helvetica,Arial' if bold else 'Helvetica,Arial'
        o.append(f'<text x="{pad}" y="{hy}" font-family="{fam}" font-size="{fsz}"{fw} fill="{fill}">{esc(seg)}</text>')
    ycur = head_h
    for r in range(rows):
        maxh = ch_per_row[r]
        for c in range(cols):
            ci = r * cols + c
            if ci >= len(card_html): continue
            x = pad + c * (cw + pad)
            title, big, wrapped, acc = card_html[ci]
            o.append(f'<rect x="{x}" y="{ycur}" width="{cw}" height="{maxh}" rx="14" fill="url(#card)" stroke="{acc}" stroke-opacity="0.35"/>')
            ty = ycur + title_fs + 8
            ttxt = title.upper()
            tfs = 15.5
            while tw(ttxt, tfs, True) > cw - 36 and tfs > 11.5: tfs -= 0.5
            if tw(ttxt, tfs, True) > cw - 36:
                while tw(ttxt + '…', tfs, True) > cw - 36 and len(ttxt) > 12: ttxt = ttxt[:-2]
                ttxt += '…'
            o.append(f'<text x="{x+18}" y="{ty}" font-family="Helvetica,Arial" font-size="{tfs}" font-weight="bold" fill="{acc}">{esc(ttxt)}</text>')
            if big:
                ty += big_fs + 2
                o.append(f'<text x="{x+18}" y="{ty}" font-family="Helvetica Neue,Helvetica,Arial" font-size="{big_fs}" font-weight="bold" fill="#f8fafc">{esc(big)}</text>')
            ty += 12
            for seg in wrapped:
                ty += lead
                o.append(f'<text x="{x+18}" y="{ty}" font-family="Helvetica,Arial" font-size="{body_fs}" fill="#cbd5e1">{esc(seg)}</text>')
        ycur += maxh + pad
    if footer:
        ft = footer[0]['t']
        ffs = 14.5
        while tw(ft, ffs) > W - pad*2 and ffs > 10.5:
            ffs -= 0.5
        if tw(ft, ffs) > W - pad*2:
            while tw(ft + '…', ffs) > W - pad*2 and len(ft) > 20: ft = ft[:-4].rstrip(' ·,')
            ft += '…'
        o.append(f'<text x="{pad}" y="{H-24}" font-family="Helvetica,Arial" font-size="{ffs}" fill="#64748b">{esc(ft)}</text>')
    o.append('</svg>')
    open(out, 'w').write('\n'.join(o))
    return len(cards)

LOGO = {'Q', 'QWEGN', 'Qwegn', 'INFERENCE ENGINE'}

def rescale_fonts(path, min_fs=18.5):
    """Chart files: keep vector geometry; scale text up per-block (logo untouched)."""
    s = open(path).read()
    vb = [float(v) for v in re.search(r'viewBox="([\d. ]+)"', s).group(1).split()]
    def bump_block(m):
        block = m.group(0)
        body = re.sub(r'<[^>]+>', '', block)
        content = html.unescape(body).strip()
        if content in LOGO:
            if content == 'INFERENCE ENGINE':
                return re.sub(r'(font-size=")([\d.]+)(")', lambda z: z.group(1) + '11' + z.group(3), block)
            return block
        block = re.sub(r'(font-size=")([\d.]+)(")', lambda z: z.group(1) + str(max(float(z.group(2)), min_fs)) + z.group(3), block)
        return block
    s = re.sub(r'<text\b[^>]*>.*?</text>', bump_block, s, flags=re.S)
    def bump_style(m):
        return re.sub(r'(font-size:)([\d.]+)px', lambda z: z.group(1) + str(max(float(z.group(2)), min_fs)) + 'px', m.group(0))
    s = re.sub(r'<style[^>]*>.*?</style>', bump_style, s, flags=re.S)
    # collision + bounds resolution: solve font sizes directly, then re-check
    def boxes_of(items):
        out = []
        for bx0, bx1, y, fs, txt, x, anchor in items: out.append((bx0, bx1))
        return out
    for _ in range(8):
        _, _, tt = parse(s)
        items = []
        for t in tt:
            w = tw(t['t'], t['fs'])
            bx0 = t['x'] if t['anchor'] == 'start' else (t['x'] - w/2 if t['anchor'] == 'middle' else t['x'] - w)
            items.append([bx0, bx0 + w, t['y'], t['fs'], t['t'], t['x'], t['anchor'], 0.0])
        moved = False
        def set_fs(it, fs):
            w2 = tw(it[4], fs)
            it[0] = it[5] if it[6] == 'start' else (it[5] - w2/2 if it[6] == 'middle' else it[5] - w2)
            it[1] = it[0] + w2; it[3] = fs
        def voverlap(a, b):
            ya, yb = a[2] + a[7], b[2] + b[7]
            return min(ya + a[3]*0.24, yb + b[3]*0.24) - max(ya - a[3]*0.78, yb - b[3]*0.78)
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j: continue
                a, b = items[i], items[j]
                if voverlap(a, b) <= 2: continue          # different visual rows
                if a[0] < b[1] - 1 and b[0] < a[1] - 1 and a[3] <= b[3]:
                    need = (b[1] - 1) - a[0]              # room a has before hitting b
                    ratio = max(0.3, need / max(a[1] - a[0], 1))
                    nfs = a[3] * ratio - 0.5
                    if nfs >= 15.5:
                        if nfs < a[3] - 0.2: set_fs(a, nfs); moved = True
                    elif a[7] == 0.0:                     # no room: stagger below instead of shrinking
                        a[7] = voverlap(a, b) + 6; moved = True
        for it in items:
            if it[0] < -2 or it[1] > vb[2] + 2:
                room = (vb[2] - 4 - it[0]) if it[1] > vb[2] + 2 else (it[1] - it[0])
                ratio = max(0.3, room / max(it[1] - it[0], 1))
                nfs = max(14.0, it[3] * ratio - 0.5)
                if nfs < it[3] - 0.2: set_fs(it, nfs); moved = True
        if not moved: break
    # final pass: stagger still-colliding boxes by shifting the lighter one down
    def voverlap(a, b):
        ya, yb = a[2] + a[7], b[2] + b[7]
        return min(ya + a[3]*0.24, yb + b[3]*0.24) - max(ya - a[3]*0.78, yb - b[3]*0.78)
    for _ in range(4):
        shifted = False
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j: continue
                a, b = items[i], items[j]
                if voverlap(a, b) > 2 and a[0] < b[1] - 1 and b[0] < a[1] - 1:
                    lighter = a if (a[1]-a[0]) <= (b[1]-b[0]) else b
                    if lighter[7] == 0.0:
                        lighter[7] = voverlap(a, b) + 6; shifted = True
        if not shifted: break
    # write resolved sizes back by absolute tag offsets
    tags = list(re.finditer(r'<text\b([^>]*)>', s))
    if len(tags) == len(items):
        edits = []
        for tag_m, it in zip(tags, items):
            a = tag_m.group(1)
            cur = re.search(r'font-size="([\d.]+)"', a)
            if cur and abs(float(cur.group(1)) - it[3]) > 0.01:
                edits.append((tag_m.start(1) + cur.start(1), tag_m.start(1) + cur.end(1), f'{it[3]:.1f}'))
            cy = re.search(r'y="([\d.]+)"', a)
            if cy and abs(float(cy.group(1)) - (float(cy.group(1)) + it[7])) > 0.5:
                edits.append((tag_m.start(1) + cy.start(1), tag_m.start(1) + cy.end(1), f'{float(cy.group(1)) + it[7]:.1f}'))
        for (s0, s1, rep) in sorted(edits, key=lambda e: -e[0]):
            s = s[:s0] + rep + s[s1:]
    open(path, 'w').write(s)

def render_ttft(path):
    """Rebuild the paired A/B TTFT bar chart from its extracted values."""
    s = open(path).read()
    vb = [float(v) for v in re.search(r'viewBox="([\d. ]+)"', s).group(1).split()]
    tx = []
    stack=[(0.,0.)]
    for m in re.finditer(r'(<g\b[^>]*>|</g>|<text\b[^>]*>.*?</text>)', s, re.S):
        tok=m.group(1)
        if tok.startswith('</g'):
            if len(stack)>1: stack.pop()
            continue
        if tok.startswith('<g'):
            dx,dy=stack[-1]; t=re.search(r'translate\(([-\d.]+)[, ]+([-\d.]+)\)',tok)
            stack.append((dx+float(t.group(1)) if t else dx, dy+float(t.group(2)) if t else dy)); continue
        a=re.match(r'<text([^>]*)>',tok).group(1)
        t=html.unescape(re.sub(r'<[^>]+>','',tok[len("<text"+a):])).strip()
        if not t: continue
        x=re.search(r'x="([-\d.]+)"',a); y=re.search(r'y="([-\d.]+)"',a)
        if x and y: tx.append((float(x.group(1))+stack[-1][0], float(y.group(1))+stack[-1][1], t))
    title = next((t for _,y,t in tx if 'Time-To-First' in t), 'TTFT')
    groups = [t for _,y,t in tx if '(' in t and 'kv=' in t]
    cats   = sorted([(x,t) for x,y,t in tx if 370 < y < 385], key=lambda c:c[0])
    subs   = sorted([(x,t) for x,y,t in tx if 390 < y < 405], key=lambda c:c[0])
    vals   = [t for x,y,t in tx if re.fullmatch(r'[0-9.]+ s', t) and y > 100]
    chips  = [t for x,y,t in tx if 'FASTER' in t]
    footer = next((t for _,y,t in tx if y > 430), '')
    nums = [float(v.split()[0]) for v in vals] or [6.53,0.12,0.40,17.84,0.24,0.45]
    W,Hh,base,top,padx = 1200, 580, 430, 130, 90
    maxv = max(nums)*1.15
    COLORS = ['#f59e0b','#10b981','#22d3ee','#f59e0b','#10b981','#22d3ee']
    esc = lambda z: z.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {Hh}" width="{W}" height="{Hh}">',
       '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0b1120"/><stop offset="1" stop-color="#04060a"/></linearGradient></defs>',
       f'<rect width="{W}" height="{Hh}" fill="url(#bg)"/>',
       f'<text x="{W/2}" y="48" text-anchor="middle" font-family="Helvetica Neue,Helvetica,Arial" font-size="26" font-weight="bold" fill="#f1f5f9">{esc(title)}</text>']
    if len(groups)==2:
        o.append(f'<text x="340" y="86" text-anchor="middle" font-family="Helvetica,Arial" font-size="18" fill="#7dd3fc">{esc(groups[0])}</text>')
        o.append(f'<text x="860" y="86" text-anchor="middle" font-family="Helvetica,Arial" font-size="18" fill="#7dd3fc">{esc(groups[1])}</text>')
    for gv in range(0, 21, 5):
        y = base - gv/maxv*(base-top)
        o.append(f'<line x1="{padx}" y1="{y:.0f}" x2="{W-30}" y2="{y:.0f}" stroke="#1e293b"/>')
        o.append(f'<text x="{padx-10}" y="{y+5:.0f}" text-anchor="end" font-family="Helvetica,Arial" font-size="14" fill="#64748b">{gv} s</text>')
    o.append(f'<text x="26" y="{(top+base)/2:.0f}" font-family="Helvetica,Arial" font-size="15" fill="#94a3b8" transform="rotate(-90 34 {(top+base)/2:.0f})" text-anchor="middle">TTFT (seconds — lower is better)</text>')
    n=max(len(nums),1)
    slot=(W-padx-90)/n
    bw=min(96, int(slot*0.6))
    for i,v in enumerate(nums):
        x = padx + 60 + i*slot - bw/2
        h = v/maxv*(base-top)
        o.append(f'<rect x="{x:.0f}" y="{base-h:.0f}" width="{bw}" height="{h:.0f}" rx="6" fill="{COLORS[i%6]}" fill-opacity="0.9"/>')
        o.append(f'<text x="{x+bw/2:.0f}" y="{base-h-12:.0f}" text-anchor="middle" font-family="Helvetica Neue,Helvetica,Arial" font-size="19" font-weight="bold" fill="#f8fafc">{v} s</text>')
        if i < len(cats):
            for wi, seg in enumerate(wrap(cats[i][1], 15.5, slot-10, True)[:2]):
                o.append(f'<text x="{x+bw/2:.0f}" y="{base+24+wi*18:.0f}" text-anchor="middle" font-family="Helvetica,Arial" font-size="15.5" font-weight="bold" fill="#e2e8f0">{esc(seg)}</text>')
        if i < len(subs):
            for wi, seg in enumerate(wrap(subs[i][1], 14, slot-10)[:2]):
                o.append(f'<text x="{x+bw/2:.0f}" y="{base+60+wi*16:.0f}" text-anchor="middle" font-family="Helvetica,Arial" font-size="14" fill="#94a3b8">{esc(seg)}</text>')
    if chips:
        o.append(f'<text x="{W/2}" y="{base+86:.0f}" text-anchor="middle" font-family="Helvetica,Arial" font-size="16" fill="#34d399">{esc(" · ".join(chips))}</text>')
    if footer:
        o.append(f'<text x="{W/2}" y="{Hh-16}" text-anchor="middle" font-family="Helvetica,Arial" font-size="14.5" fill="#64748b">{esc(footer)}</text>')
    o.append('</svg>')
    open(path,'w').write('\n'.join(o))

def _main():
    src = sys.argv[1] if len(sys.argv) > 1 else 'assets'
    for p in sorted(glob.glob(os.path.join(src, 'qwegn_*.svg'))):
        name = os.path.basename(p)[:-4]
        if 'lsa_scaling' in name:
            rescale_fonts(p); print(f'{name}: fonts rescaled'); continue
        if 'ttft_ab_chart' in name:
            render_ttft(p); print(f'{name}: chart rebuilt'); continue
        vb, rects, texts = parse(p)
        header, panels, leftover, footer = assign(vb, rects, texts)
        if panels or leftover or header:
            render(name, vb, header, panels, leftover, footer, p)
            print(f'{name}: {len(panels)} cards rebuilt')


if __name__ == '__main__':
    _main()
