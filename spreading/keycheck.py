"""Statement readers shared by the corrections, the grader and the answer-key checker:
FilingSummary's statement R pages, the inline instance's facts, and the amounts an R page prints."""
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from decimal import Decimal as D
from pathlib import Path

X = '{http://www.xbrl.org/2003/instance}'


def statement_pages(fil, tpl):
    """{statement: R page}: the one Statements report of FilingSummary the template's own rules choose
    (template.find_statement, the pipeline's choice: title words and exclusions, else the title fallback, by the
    concepts the page prints). Problems for none or several."""
    from . import template
    reports = [(r.findtext('HtmlFileName'), (r.findtext('LongName') or '') + ' ' + (r.findtext('ShortName') or ''))
               for r in ET.parse(fil / 'FilingSummary.xml').getroot().iter('Report')
               if r.findtext('MenuCategory') == 'Statements']
    pages, problems = {}, []
    for st in ('IS', 'BS', 'CF'):
        hit, how, why = template.find_statement(tpl['statements'][st], reports, lambda n: page_concepts(fil / n))
        if len(hit) != 1:
            problems.append(f'{st}: FilingSummary has {len(hit)} statement pages by the template\'s rules: {hit}; {why}')
        else:
            pages[st] = hit[0]
    return pages, problems


def page_concepts(page):
    """The concepts an R page prints a row for (its defref anchors), as prefix:name."""
    return {f'{a}:{b}' for a, b in re.findall(r"defref_([^_'\"]+)_([^'\"]+)", page.read_text(encoding='utf8'))}


def load_instance(fil):
    """(facts by id, facts by concept, contexts, the cover page's fiscal period) from the one extracted instance."""
    inst = list(fil.glob('*_htm.xml'))
    if len(inst) != 1:
        raise SystemExit(f'{fil}: {len(inst)} extracted instances (*_htm.xml), not one')
    uri2pre = {}
    for ev, (pre, uri) in ET.iterparse(inst[0], events=['start-ns']):
        uri2pre.setdefault(uri, pre)
    root = ET.parse(inst[0]).getroot()
    ctx = {}
    for c in root.iter(f'{X}context'):
        p = c.find(f'{X}period')
        i = p.find(f'{X}instant')
        per = i.text if i is not None else f"{p.find(f'{X}startDate').text}/{p.find(f'{X}endDate').text}"
        seg = c.find(f'{X}entity/{X}segment')
        ctx[c.get('id')] = (per, seg is not None and len(seg) > 0, c.find(f'{X}entity/{X}identifier').text)
    facts, by_concept, cover = {}, defaultdict(list), None
    for el in root:
        if el.get('contextRef') is None:
            continue
        uri, local = el.tag[1:].split('}')
        f = {'concept': f'{uri2pre.get(uri, uri)}:{local}', 'context': el.get('contextRef'), 'unit': el.get('unitRef'),
             'decimals': el.get('decimals'), 'value': el.text}
        if el.get('id'):
            facts[el.get('id')] = f
        by_concept[f['concept']].append(f)
        if local == 'DocumentPeriodEndDate' and not ctx[f['context']][1]:
            cover = ctx[f['context']][0]
    return facts, by_concept, ctx, cover


def unit_kinds(fil):
    """{unit id: 'money' | 'shares' | 'other'} from the instance's unit definitions (a divide, such as per share, and
    pure are 'other')."""
    inst = next(Path(fil).glob('*_htm.xml'))
    kinds = {}
    for u in ET.parse(inst).getroot().iter(f'{X}unit'):
        ms = [m.text.split(':')[-1] for m in u.iter(f'{X}measure')]
        kinds[u.get('id')] = ('other' if u.find(f'{X}divide') is not None or len(ms) != 1 else
                              'shares' if ms[0] == 'shares' else
                              'money' if re.fullmatch(r'[A-Z]{3}', ms[0]) else 'other')
    return kinds


def page_value(fact, kinds, scale, share_scale):
    """A fact's value as the page prints it: money at the page's amount scale, shares at its share scale, others as is."""
    k = kinds.get(fact['unit'])
    return abs(D(fact['value'])) / (scale if k == 'money' else share_scale if k == 'shares' else 1)


MONTHS = {m: i for i, m in enumerate(('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov',
                                      'dec'), 1)}
SCALE = {'Thousands': D(1000), 'Millions': D(10) ** 6, 'Billions': D(10) ** 9}


def header_date(text):
    """The date of a column heading: its first non-empty line, which must be a whole date (a cell may print the
    unit on a second line)."""
    first = next((t.strip() for t in text.splitlines() if t.strip()), '')
    m = re.fullmatch(r'([A-Za-z]{3})[a-z]*\.? (\d{1,2}), (\d{4})', first)
    return f'{m.group(3)}-{MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}' if m else None


def printed_amounts(page, end, problems):
    """(Counter of (concept, printed amount), money scale, share scale) for the page's column of the fiscal year
    ending `end`: the column dated `end`, under a twelve-month (or 52/53-week) heading when the page also prints
    shorter periods (quarterly columns come first on some pages)."""
    rows, scale, share_scale = page_rows(page, end, problems)
    return Counter((c, a) for c, _, a in rows), scale, share_scale


def page_rows(page, end, problems, signed=False):
    """([(concept, member header, printed amount)], money scale, share scale), in printed order, for the column of
    printed_amounts. The member header is the member block (`<tr class="rh">`) a row sits under, (defref
    'axis=member', printed text), None above the first block; amounts are absolute, at the page's scale, or with
    `signed` an amount printed in parentheses negative (the split's binding, corrections.split_of)."""
    html = page.read_text(encoding='utf8')
    title = re.sub('<[^>]+>', ' ', re.search(r'<th class="tl"[^>]*>(.*?)</th>', html, re.S).group(1))
    sh = re.search(r'shares in (Thousands|Millions|Billions)', title)
    share_scale = SCALE[sh.group(1)] if sh else D(1)
    rest = re.sub(r'shares in (Thousands|Millions|Billions)', '', title)
    m = re.search(r'in (Thousands|Millions|Billions)', rest)
    scale = SCALE[m.group(1)] if m else D(1)
    heads = re.findall(r'<tr[^>]*>((?:(?!</tr>).)*?<th class="th".*?)</tr>', html, re.S)
    # a heading's </div> and <br> are line breaks, so a date on the cell's first line is found
    ths = [[(int(c or 1), re.sub('<[^>]+>', '', re.sub(r'</div>|<br\s*/?>', '\n', t, flags=re.I)).strip()) for c, t in
            re.findall(r'<th class="th"(?: colspan="(\d+)")?[^>]*>(.*?)</th>', h, re.S)] for h in heads]
    slots = [(n, t) for c, t in ths[-1] for n in [t] * c] if ths else []
    groups = [' '.join(t.split()) for c, t in ths[0] for _ in range(c)] if len(ths) > 1 else [''] * len(slots)
    cand = [i for i, (d, _) in enumerate(slots) if header_date(d) == end]
    if len(cand) > 1:
        cand = [i for i in cand if re.search(r'12 Months|5[23] Weeks', groups[i] if i < len(groups) else '')]
    if len(cand) != 1:
        problems.append(f'{page.name}: no single column for the year ending {end} ({len(cand)} found)')
        return [], scale, share_scale
    col, block, out = cand[0], None, []
    for cls, tr in re.findall(r'<tr class="(r[a-z]*)">(.*?)</tr>', html, re.S):
        if cls == 'rh':
            m = re.search(r"defref_([^'\"]+)", tr)
            text = re.sub(r'<[^>]+>|&#160;', ' ', re.search(r'<td[^>]*>(.*?)</td>', tr, re.S).group(1))
            block = (m.group(1), ' '.join(text.split())) if m else None
            continue
        m = re.search(r"defref_([^_'\"]+)_([^'\"]+)", tr)       # local names may hold hyphens
        cells = [c for c in re.findall(r'<td class="([^"]*)"[^>]*>(.*?)</td>', tr, re.S)
                 if not c[0].startswith('pl') and c[0] != 'th']        # after the label cell and a footnote-marker
        # column beside it (a two-column label heading); footnote cells kept
        if not m or len(cells) <= col or cells[col][0] not in ('nump', 'num'):
            continue
        t = re.sub(r'<[^>]+>|&#160;|\s|\$', '', cells[col][1])
        if not t or t.strip('()').endswith('%'):      # a rate among the amounts, "5.89%" or "(5.89%)"
            continue
        v = D(t.strip('()').replace(',', ''))
        out.append((f'{m.group(1)}:{m.group(2)}', block, -v if signed and t.startswith('(') else v))
    return out, scale, share_scale
