"""Frozen Strategy V2 representation rules (2026-11-26).

Prediction and reference use identical rules. No model calls, gold-guided
extraction, coordinate repair, unit conversion or changes to official formulas.
This standalone snapshot intentionally has no experiment-runner dependencies.
"""
import ast
import copy
import json
import re
import unicodedata
from decimal import Decimal

VERSION = 'strategy_v2_20261126_units_v2'


def same_value(left, right):
    return json.dumps(left, sort_keys=True, ensure_ascii=False, allow_nan=False) == json.dumps(right, sort_keys=True, ensure_ascii=False, allow_nan=False)

def pairs(items):
    result = {}
    for key, value in items:
        if key in result and not same_value(result[key], value):
            raise ValueError("Conflicting duplicate JSON key: " + key)
        result[key] = value
    return result

def constant(value):
    raise ValueError("Non-standard JSON constant: " + value)

DECODER = json.JSONDecoder(object_pairs_hook=pairs, parse_constant=constant)

def _object_spans(text):
    """Skip complete outer objects/arrays, including malformed JSON; never fish inside them."""
    i = 0
    while i < len(text):
        if text[i] not in "{[":
            i += 1
            continue
        start, stack, quoted, escaped = i, [], False, False
        while i < len(text):
            char = text[i]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack or (stack[-1], char) not in (("{", "}"), ("[", "]")):
                    break
                stack.pop()
                if not stack:
                    i += 1
                    yield start, i
                    break
            i += 1
        else:
            return
        if stack:
            # A damaged outer object is not a license to reuse its nested fragments.
            return

def normalize_structured_text(text):
    if not isinstance(text,str) or not re.match(r'^\s*(?:\[\s*)?\{\\"',text):
        return text,[]
    try:
        decoded=json.loads('"'+text.replace('\n',r'\n')+'"')
        spans=list(_object_spans(decoded))
        if not spans:return text,[]
        end=0
        for start,stop in spans:
            if decoded[end:start].strip(' \r\n\t;,'):return text,[]
            raw=decoded[start:stop]
            try:DECODER.decode(raw)
            except ValueError:
                tree=ast.parse(raw,mode='eval')
                for node in ast.walk(tree):
                    if isinstance(node,ast.Dict):
                        keys=[ast.literal_eval(k) for k in node.keys]
                        if len(keys)!=len(set(keys)):raise ValueError('duplicate keys')
                ast.literal_eval(tree)
            end=stop
        if decoded[end:].strip():return text,[]
        return decoded,['decode_one_extra_structured_escape_layer']
    except (ValueError,SyntaxError,TypeError):
        return text,[]

def objects(text):
    """Extract balanced literal dictionaries without executing model text."""
    out, start, depth, quote, escaped = [], None, 0, None, False
    for i, char in enumerate(str(text)):
        if start is None:
            if char == '{': start, depth, quote, escaped = i, 1, None, False
            continue
        if quote:
            if escaped: escaped = False
            elif char == '\\': escaped = True
            elif char == quote: quote = None
        elif char in ('"', "'"): quote = char
        elif char == '{': depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                literal = str(text)[start:i + 1]
                for parser in (json.loads, ast.literal_eval):
                    try:
                        value = parser(literal)
                        if isinstance(value, dict): out.append(value)
                        break
                    except (ValueError, SyntaxError, TypeError, json.JSONDecodeError): pass
                start = None
    return out

POS = {'entails','entailed','entailment','supports','supported','support','upholds','upheld',
       'confirms','confirmed','affirms','affirmed','substantiates','true','yes'}

NEG = {'refutes','refuted','refutation','contradicts','contradicted','contradiction','conflicts',
       'conflicts with','disputes','disproved','disproves','opposes','opposed','challenged','false','no'}

NEUTRAL = {'neutral','not sure','not enough information','not enough evidence','insufficient information',
           'insufficient evidence','unknown'}

def clean(x):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(x)).replace('’', "'")).strip().casefold()

def option_map(question):
    matches = list(re.finditer(r'\(([A-Da-d1-4])\)\s+|(?<!\w)([A-D])\.\s+', question))
    result = {}
    for i, m in enumerate(matches):
        text = question[m.end():matches[i+1].start() if i+1<len(matches) else len(question)]
        text = re.split(r'\n\s*\n|\n(?:Your|Solve|Find|Select|Output|Answer)\b',text)[0].strip(' ;\n.')
        result[(m[1] or m[2]).casefold()] = clean(text)
    return result if len(result)>=2 else {}

def extract(text):
    """Read explicit answer fields; reject competing fields instead of picking the gold one."""
    found = [o['answer'] for o in objects(text) if 'answer' in o]
    if not found:
        if text.strip() and not any(c in text for c in '{}'):
            return [text.strip()], ['plain_answer']
        return None, ['unparseable_or_empty']
    if any(v != found[0] for v in found):
        return None, ['conflicting_answer_fields']
    value = found[0]; rules = []
    if isinstance(value, str):
        # Only recover an explicitly marked, terminal final-answer object, not a number in reasoning.
        unescaped = value.replace('\\"', '"')
        m = re.search(r'\bfinal answer\s*(?:is|:)?\s*(\{.*\})[.\s]*$',unescaped,re.I|re.S)
        if m:
            nested = [o['answer'] for o in objects(m[1]) if 'answer' in o]
            if len(nested)==1 and isinstance(nested[0],(str,int,float,list)):
                value=nested[0];rules.append('explicit_nested_final_answer')
    if isinstance(value,(str,int,float)) and not isinstance(value,bool): return [str(value)],rules
    if isinstance(value,list) and all(isinstance(v,(str,int,float)) and not isinstance(v,bool) for v in value):
        return [str(v) for v in value],rules
    return None,rules+['unsupported_answer_type']

def context(question, table):
    rows=table.get('table_rows',[]);blob=clean(json.dumps(rows,ensure_ascii=False))
    aliases={}
    # Source-attested country abbreviations; never remove arbitrary parentheses.
    for row in rows:
        for cell in row:
            m=re.fullmatch(r'(.+?)\s*\(([A-Z]{2,4})\)',str(cell))
            if m:aliases[clean(cell)]=clean(m[1])
        # A parenthetical employee count is supplemental only when the source row supports it.
        if row and 'employee' in clean(question):
            for cell in row[1:]:
                if re.fullmatch(r'\d[\d,]*',str(cell)):
                    aliases[clean(str(row[0])+f' (approximately {cell} employees)')]=clean(row[0])
    return {'question':question,'options':option_map(question),'table_text':blob,'aliases':aliases}

def canonical(value, ctx, tfv=False):
    original=value;text=clean(value).rstrip('.');rules=[]
    opts=ctx['options']
    prefix=re.match(r'^(?:\(([a-d1-4])\)|([a-d])\.)\s*(.*)$',text)
    if text in opts:
        text=opts[text];rules.append('choice_label_to_text')
    elif prefix and (prefix[1] or prefix[2]) in opts:
        label=prefix[1] or prefix[2];body=prefix[3]
        if body==opts[label]:text=body;rules.append('choice_prefix')
        else:return text,['choice_label_content_conflict']
    if tfv:
        # Strip only a standard truth-condition gloss or explicit opening verdict.
        text=re.sub(r'\s*\(the sentence is (?:true|false) given the table\)$','',text)
        m=re.match(r'^the table ([a-z ]+?) the (?:sentence|statement|hypothesis)\.',text)
        if m:text=m[1];rules.append('explicit_opening_verdict')
        for label,variants in [('entailed',POS),('refuted',NEG),('neutral',NEUTRAL)]:
            if text in variants:return label,rules+(['relation_synonym'] if text!=label else [])
        return text,rules
    if text in ctx['aliases']:text=ctx['aliases'][text];rules.append('source_attested_annotation')
    blob=ctx['table_text'];q=clean(ctx['question'])
    # Table-specific units are accepted only with explicit source evidence.
    if re.fullmatch(r'[+-]?[\d,.]+ \(cents per asm\)',text) and '¢' in blob and 'asm' in q:
        text=text.removesuffix(' (cents per asm)');rules.append('source_attested_unit')
    if re.fullmatch(r'[+-]?[\d,.]+ \(000s\)',text) and re.search(r'asms?\).*?\(000s\)',blob):
        text=text.removesuffix(' (000s)');rules.append('source_attested_scale')
    if re.fullmatch(r'[+-]?[\d,.]+%',text) and re.search(r'percent(?:age)?',q):
        text=text[:-1];rules.append('question_percent_unit')
    if text.startswith('$') and ('$' in blob or '$' in q) and re.fullmatch(r'\$[+-]?[\d,.]+',text):
        text=text[1:];rules.append('source_attested_currency')
    # Accounting negatives remain negative; do not discard signs, scales, percent or arbitrary units.
    if re.fullmatch(r'\([\d,.]+\)',text):text='-'+text[1:-1];rules.append('accounting_negative')
    if re.fullmatch(r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?',text):
        text=format(Decimal(text.replace(',','')),'f')
        if '.' in text:text=text.rstrip('0').rstrip('.')
        if text=='-0':text='0'
        if text!=clean(original):rules.append('numeric_spelling')
    return text,rules

NUMBER=r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'

def scalar_unit(value,ctx):
    text=' '.join(str(value).casefold().split()).rstrip('.')
    q=ctx['question'].casefold()
    # Restrict to scalar lookup. Rates, changes and derived quantities require
    # their own dimensions and must not inherit a level row's unit blindly.
    if re.search(r'\b(change|changed|growth|grow|grew|increase|decrease|difference|percent|percentage|ratio)\b',q):return None
    try:rows=json.loads(ctx['table_text'])
    except (ValueError,TypeError):return None
    candidates=[]
    for row in rows:
        for cell in row:
            label=str(cell).casefold()
            units=re.findall(r'\((miles|kilometers|hours|thousands|millions|billions)\)',label)
            if len(units)!=1:continue
            metric=re.sub(r'\([^)]*\)','',label).strip()
            aliases=re.findall(r'\(["\']([a-z]+)["\']\)',label)
            matches=lambda a:bool(re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',q))
            if not ((len(metric.split())>=2 and matches(metric)) or any(matches(a) or (a.endswith('s') and matches(a[:-1])) for a in aliases)):continue
            candidates.append((units[0],metric,aliases))
    if not candidates or len({x[0] for x in candidates})!=1:return None
    unit=candidates[0][0];scale=unit in ('thousands','millions','billions')
    suffixes={unit,unit[:-1]}
    if scale:
        for _,metric,aliases in candidates:
            for name in [metric,*aliases]:
                suffixes.update({unit+' '+name,unit[:-1]+' '+name})
    m=re.fullmatch('('+NUMBER+r')\s+(.+?)(?:\s*\(('+NUMBER+r')\))?',text)
    if not m or m[2] not in suffixes:return None
    num=Decimal(m[1].replace(',',''))
    if m[3] is not None and Decimal(m[3].replace(',',''))!=num:return None
    return m[1]

GROUPS = {
 'entailed': 'support|supports|supported|entail|entails|entailed|substantiate|substantiates|substantiated|confirm|confirms|confirmed|affirm|affirms|affirmed|uphold|upholds|upheld|validate|validates|validated',
 'refuted': 'contradict|contradicts|contradicted|refute|refutes|refuted|disprove|disproves|disproved|dispute|disputes|disputed|challenge|challenges|challenged|oppose|opposes|opposed|conflicts with',
 'neutral': 'neutral|not enough information|not enough evidence|not sure',
}

ALIASES = {v: k for k, values in GROUPS.items() for v in values.split('|')}

def normalize_label(value):
    text = ' '.join(unicodedata.normalize('NFKC', str(value)).lower().split()).rstrip('.').strip()
    if text in ALIASES:
        return ALIASES[text]
    # Only an entire explicit verdict optionally wrapped in an option marker and
    # a matching truth gloss. Never search prose for a positive/negative word.
    match = re.fullmatch(r'(?:\([a-z]\)\s*)?([a-z ]+?)(?:\s*\((?:the )?(?:claim|statement|sentence|hypothesis) is (true|false)\))?', text)
    if match:
        label, gloss = match.groups()
        category = ALIASES.get(label.strip())
        if category and (not gloss or category == {'true': 'entailed', 'false': 'refuted'}[gloss]):
            return category
    return None

def canon(value,ctx,tfv=False):
 if not tfv:
  scalar=scalar_unit(value,ctx)
  if scalar is not None:
   normalized,rules=canonical(scalar,ctx,False)
   return normalized,rules+['source_attested_scalar_unit_v2']
 if tfv:
  label=normalize_label(value)
  if label is not None:return label,['tfv_exact_label_v2']
 # November 10 v5 addendum, applied equally to all arms and references.
 if not tfv and value.strip().endswith("¢") and "¢" in ctx["table_text"] and "asm" in ctx["question"].lower():value=value.strip()[:-1]
 return canonical(value,ctx,tfv)

def text_payload(text):
 text=re.sub(r'^\s*```(?:json|text|markdown)?\s*\n','',text)
 text=re.sub(r'\n```\s*$','',text).strip()
 try:
  obj=json.loads(text)
  if isinstance(obj,dict) and isinstance(obj.get('answer'),str) and set(obj)<={'answer','key_evidence','uncertainties'}:text=obj['answer']
 except ValueError:pass
 return re.split(r'\n\s*(?:key_evidence|uncertainties)\s*:',text,maxsplit=1)[0].strip()

def mcd_text(text):
 s=text.strip()
 if s.startswith('```') and s.endswith('```'):
  s=s.split('\n',1)[-1].rsplit('```',1)[0].strip()
 try:obj=json.loads(s)
 except (ValueError,TypeError):
  try:obj=ast.literal_eval(s)
  except (ValueError,SyntaxError):return text
 if isinstance(obj,dict) and set(obj)=={'answer'}:
  obj=obj['answer']
  if isinstance(obj,str):return mcd_text(obj)
 if not isinstance(obj,list):return text
 if not obj:return 'The table does not contain any merged cells.'
 converted=[]
 for region in obj:
  if not isinstance(region,dict) or set(region)!={'top-left','bottom-right'}:return text
  d={}
  for k,v in region.items():
   if isinstance(v,str):
    try:v=ast.literal_eval(v)
    except (ValueError,SyntaxError):return text
   if not isinstance(v,(tuple,list)) or len(v)!=2 or any(type(i) is not int for i in v):return text
   d[k]=tuple(v)
  converted.append(d)
 return repr(converted)

def records_base(text):
    def walk(value):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from walk(child)
    return [record for value in objects(text) for record in walk(value)]


def qa_wire(values):
 # Protect literal punctuation from the official regex's quote/bracket substitutions.
 # ast.literal_eval reconstructs exactly the same strings; no answer-content change.
 encoded=[]
 for v in values:
  token=json.dumps(v,ensure_ascii=True).replace(chr(92)+chr(34),r'\u0022').replace("'",r'\u0027').replace('[',r'\u005b').replace(']',r'\u005d')
  assert ast.literal_eval(token)==v
  encoded.append(token)
 return '{"answer": ['+', '.join(encoded)+']}'

def coord(v):
 if isinstance(v,str):
  try:v=ast.literal_eval(v)
  except (ValueError,SyntaxError):return v
 if isinstance(v,(list,tuple)) and len(v)==2 and all(type(x)is int for x in v):return str(tuple(v))
 return v

def normalize_structure(task,text):
 # Read explicit structured fields only; no reference-dependent field/coordinate repair.
 if task=='MCD':return mcd_text(text)
 text,_=normalize_structured_text(text);objs=records_base(text);out=[]
 for o in objs:
  if task.endswith('TSD') and {'row_number','column_number'}<=o.keys():
   out.append({k:str(o[k]) for k in ['row_number','column_number']})
  elif task.endswith('TCL') and {'value','location'}<=o.keys():
   out.append({'value':clean(o['value']),'location':coord(o['location'])})
  elif task.endswith('TCE') and {'row_id','column_id','cell_value'}<=o.keys():
   out.append({'row_id':o['row_id'],'column_id':o['column_id'],'cell_value':clean(o['cell_value'])})
  elif task.endswith('RCE') and 'cell_list' in o and isinstance(o['cell_list'],list):
   axes=[k for k in ['row_id','column_id'] if k in o]
   if len(axes)==1:out.append({axes[0]:o[axes[0]],'cell_list':[clean(v) for v in o['cell_list']]})
 if not out:return text
 if task.endswith('TSD'):
  if any(x!=out[0] for x in out):return text
  return repr(out[0])
 return '\n'.join(repr(v) for v in out)

SOURCE_SHA256 = {'table_tools/strict_json.py': 'e57eeaabe0391d25e2fef02f34f89e64acbbc5d11abe2dc48b0e135900514f20', 'tool_loop_runtime/wire_json.py': 'f0fa3ca2d98b12d7496a57812225a0c80715d9c7d20bdadacf08492a5190a801', 'tool_loop_runtime/answer_text.py': '45e1c4f4060abd467ccc2336e0fc18de49723043f97e2140a52e541b52891a28', 'scripts/analyze_glm_eval200.py': '2056e3f61ab2ffe9eb553f5113e6f8aab2bd753b743edd391c2b221563691aba', 'scripts/score_semantic500.py': 'bd2275e8871ed4a4425c734547a41b98802d4278cb18c9c122005fe670601720', 'ep_strategy/qa_unit_normalization.py': '786cf63bdbc2451dfd97a2cd7402ba525230d1165585d8b59269e4fd809d5a3e', 'ep_strategy/tfv_normalization.py': 'c1a43131b888c775164e5117d82d96a463fb3163b262c47df95d34f1e1f86cb4', 'ep_strategy/analyze_eval500_v18.py': '46f44fd85658631d136f0ba020958e7bf085a42a9229b7796c1f6a9fde68c574', 'ep_strategy/normalize_eval500_main_test.py': '71167fc53950e411a3c12aec002ef22ca998c3476255432c22c0c40ea89d9dd0'}

def normalize_group(task, benchmark, rows, gold, tables):
    """Return scoring copies plus a per-sample audit; leave inputs untouched."""
    refs, table_refs, converted, details = {}, {}, [], []
    for row in rows:
        sid = row['question_id']
        item = copy.deepcopy(gold[sid])
        raw = row['text']
        answer = raw
        rules = []
        reference_before = copy.deepcopy(item.get('answer_list'))
        qa = task in ('TQA', 'TFV') and benchmark != 'FeTaQA'
        if qa:
            table = tables.get(item.get('image_id'), {})
            ctx = context(item.get('input', ''), table)
            item['answer_list'] = [canon(str(v), ctx, task == 'TFV')[0] for v in item['answer_list']]
            values, rules = extract(raw)
            if values:
                canonicalized = [canon(v, ctx, task == 'TFV') for v in values]
                answer = qa_wire([v for v, _ in canonicalized])
                rules += [rule for _, applied in canonicalized for rule in applied]
        elif task == 'TR':
            pass
        elif task == 'T2T' or benchmark == 'FeTaQA':
            answer = text_payload(raw)
        else:
            item['output'] = normalize_structure(task, item['output'])
            answer = normalize_structure(task, raw)
        image_id = item.get('image_id')
        if image_id in tables and image_id not in table_refs:
            table = tables[image_id]
            table_refs[image_id] = dict(table, table_rows=[[clean(v) for v in r] for r in table['table_rows']])
        refs[sid] = item
        converted.append(dict(row, text=answer))
        details.append({'id': sid, 'task': task, 'benchmark': benchmark,
                        'raw': raw, 'normalized': answer, 'changed': raw != answer,
                        'rules': rules, 'reference_before': reference_before,
                        'reference_after': item.get('answer_list'),
                        'reference_output_before': gold[sid].get('output') if not qa else None,
                        'reference_output_after': item.get('output') if not qa else None})
    return converted, refs, table_refs, details
