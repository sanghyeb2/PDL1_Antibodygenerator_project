"""LoRA 통합 실행: 설정 -> 분할 -> 학습 -> 평가/생성 -> 시각화.

그냥 실행: 학습된 최적 LoRA로 PD-L1 항체 1쌍 생성
python LoRA.py --mode summary       : 기존 본학습 결과 확인 (CPU)
python LoRA.py --mode plot --show   : 기존 결과 시각화 (CPU)
python LoRA.py --mode train         : 새 어댑터 학습 + 최적 결과 시각화
python LoRA.py --mode evaluate      : 최적 LoRA의 검증 NLL 평가
python LoRA.py --mode split         : 새 데이터 분할
python LoRA.py --mode check         : 경로/의존성 확인

이 파일은 MAGE.py나 이전 개별 실행 파일을 import하지 않는다.
외부 ProGen 모델과 설치된 Python 패키지만 사용한다.
"""
import argparse
import csv
import traceback
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

# ── 1. 여기서 설정을 바꾸세요 ─────────────────────────────────────
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHECKPOINT = ROOT/'MAGE_ab_generation/model_epoch4'
SOURCE = HERE/'외부코드/progen2'
PYTHON = ROOT/'MAGE_ab_generation/.venv-mage-gpu/Scripts/python.exe'
DEFAULT_DATA = ROOT/'datasets/PDL1_최종_튜닝용_실측양성_106쌍.csv'
DEFAULT_SPLIT = HERE/'학습결과/데이터분할_시드42'
DEFAULT_RUN = HERE/'학습결과/본학습_20261108_시드42'
DEFAULT_ADAPTER = DEFAULT_RUN/'최적어댑터'
CONFIG = dict(mode='generate', seed=42, top_p=0.9, temperature=1.0, max_new_tokens=256)
TRAIN_CONFIG = dict(rank=8, alpha=16, dropout=0.05, learning_rate=1e-4,
                    weight_decay=0.01, accumulation=8, max_epochs=20,
                    patience=4, min_delta=0.001, max_grad_norm=1.0)
SPLIT_CONFIG = dict(seed=42, threshold=0.90, val_fraction=0.20)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HOME'] = str(HERE/'캐시/huggingface')
os.environ['TRANSFORMERS_CACHE'] = str(HERE/'캐시/huggingface/hub')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'캐시/matplotlib'))
AA = set('ACDEFGHIKLMNPQRSTVWY')


# ── 2. 실행환경·모델 로딩·생성·NLL 평가 ───────────────────────────
def use_verified_python(seed):
    """일반 Python으로 실행해도 기존 가상환경을 자동 선택한다. 설치는 하지 않는다."""
    target = PYTHON if PYTHON.is_file() else Path(sys.executable)
    if Path(sys.executable).resolve() != target.resolve() or os.environ.get('PYTHONHASHSEED') != str(seed):
        raise SystemExit(subprocess.call([str(target),'-X','utf8','-u',str(Path(__file__).resolve()),*sys.argv[1:]],
                                        env=dict(os.environ,PYTHONHASHSEED=str(seed))))

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()

def write_json(path,value):
    destination=Path(path);temporary=destination.with_name(destination.name+'.writing.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(destination)

def new_output(label):
    stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S_%f')
    return HERE/'생성결과'/f'{label}_{stamp}'

def check_environment(extra=()):
    required=('torch','transformers','tokenizers','accelerate','safetensors')+tuple(extra)
    missing=[name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError('필요한 패키지가 없습니다: '+', '.join(missing)+'; 시작안내.md를 확인하세요.')
    if not CHECKPOINT.is_dir() or not (SOURCE/'progen/modeling_progen.py').is_file():
        raise FileNotFoundError('맨 위 CHECKPOINT / SOURCE 경로를 확인하세요.')
    return {name:importlib.metadata.version(name) for name in required}

def read_antigen(data):
    with Path(data).open(encoding='utf-8-sig',newline='') as handle:
        antigens={r['antigen_seq'] for r in csv.DictReader(handle)}
    if len(antigens)!=1:
        raise ValueError('CSV 항원이 하나가 아닙니다. --antigen으로 아미노산 서열을 지정하세요.')
    sequence=next(iter(antigens))
    if not sequence or not set(sequence)<=AA: raise ValueError('항원 아미노산 서열이 잘못됐습니다.')
    return sequence

def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def load_model(adapter=None):
    import torch
    from transformers import PreTrainedTokenizerFast
    check_environment(('peft',) if adapter is not None else ())
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('이 실행 설정은 CUDA BF16 지원 GPU가 필요합니다.')
    sys.path.insert(0,str(SOURCE))
    from progen.modeling_progen import ProGenForCausalLM
    model,info=ProGenForCausalLM.from_pretrained(str(CHECKPOINT),torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,local_files_only=True,output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('모델/가중치 불일치: '+str(info))
    if adapter is not None:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(adapter))
    tokenizer=PreTrainedTokenizerFast.from_pretrained(str(CHECKPOINT),local_files_only=True)
    if len(tokenizer)!=model.config.vocab_size: raise ValueError('토크나이저 크기 불일치')
    model.eval().requires_grad_(False)
    model.to('cuda')
    return model,tokenizer

def parse_antibody(new_ids, tokenizer):
    stop_ids={tokenizer.pad_token_id,tokenizer.eos_token_id}-{None}
    stop_position=next((i for i,t in enumerate(new_ids) if t in stop_ids),len(new_ids))
    sequence_ids=new_ids[:stop_position]
    lc_id=tokenizer.convert_tokens_to_ids('[LC]')
    vh=vl=None
    if sequence_ids.count(lc_id)==1:
        boundary=sequence_ids.index(lc_id)
        vh=tokenizer.decode(sequence_ids[:boundary],skip_special_tokens=False).strip()
        vl=tokenizer.decode(sequence_ids[boundary+1:],skip_special_tokens=False).strip()
    standard=bool(vh and vl and set(vh)<=AA and set(vl)<=AA)
    length_ok=bool(vh and vl and len(vh)>=100 and len(vl)>=100)
    return dict(termination_found=stop_position<len(new_ids),
                termination_token=tokenizer.convert_ids_to_tokens(new_ids[stop_position]) if stop_position<len(new_ids) else None,
                lc_token_count=sequence_ids.count(lc_id),vh=vh,vl=vl,
                vh_length=len(vh) if vh is not None else None,vl_length=len(vl) if vl is not None else None,
                standard_amino_acids_only=standard,
                basic_sequence_checks_passed=bool(stop_position<len(new_ids) and standard and length_ok))

def generate(args,adapter=None):
    import torch
    set_seed(args.seed)
    antigen=args.antigen or read_antigen(args.data)
    if not antigen or not set(antigen)<=AA: raise ValueError('항원 서열을 확인하세요.')
    print('모델 로딩: '+('LoRA' if adapter is not None else '원본 MAGE'),flush=True)
    start=time.perf_counter()
    model,tokenizer=load_model(adapter)
    inputs=tokenizer(antigen+'[SEP]',return_tensors='pt')
    inputs={k:v.to('cuda') for k,v in inputs.items() if k in ('input_ids','attention_mask')}
    size=inputs['input_ids'].shape[1]
    if size+args.max_new_tokens>model.config.n_positions: raise ValueError('문맥 길이를 초과합니다.')
    with torch.inference_mode():
        ids=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=True,
            top_p=args.top_p,temperature=args.temperature,pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)[0,size:].tolist()
    antibody=parse_antibody(ids,tokenizer)
    output=args.output or new_output('LoRA' if adapter is not None else 'MAGE')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(model='LoRA' if adapter is not None else 'MAGE',seed=args.seed,antigen=antigen,
                checkpoint=str(CHECKPOINT),adapter=str(adapter) if adapter is not None else None,
                dtype='bfloat16',top_p=args.top_p,temperature=args.temperature,max_new_tokens=args.max_new_tokens,
                tokenizer_sha256=sha256(CHECKPOINT/'tokenizer.json'),
                raw_ids=ids,raw_text=tokenizer.decode(ids),antibody=antibody,
                elapsed_seconds=time.perf_counter()-start,scope='서열 생성과 기본 형식 검사; 결합력 검증 아님')
    write_json(output/'result.json',result)
    if antibody['basic_sequence_checks_passed']:
        (output/'candidate.fasta').write_text('>candidate_VH\n'+antibody['vh']+'\n>candidate_VL\n'+antibody['vl']+'\n',encoding='utf-8')
    print(json.dumps(antibody,ensure_ascii=False,indent=2));print('저장: '+str(output))

def verified_validation_rows(data,split):
    """원본 CSV 해시와 행 내용/ID/그룹을 확인해 두 모델이 같은 정답을 평가하게 한다."""
    manifest=json.loads((split/'split_manifest.json').read_text(encoding='utf-8'))
    if sha256(data)!=manifest['input_sha256']: raise ValueError('원본 CSV 해시가 다릅니다.')
    with Path(data).open(encoding='utf-8-sig',newline='') as handle:
        original={r['id']:r for r in csv.DictReader(handle)}
    with (split/'validation.csv').open(encoding='utf-8-sig',newline='') as handle:
        rows=sorted(csv.DictReader(handle),key=lambda r:r['id'])
    expected={a['id']:a['group_id'] for a in manifest['assignments'] if a['split']=='validation'}
    if sorted(r['id'] for r in rows)!=sorted(expected): raise ValueError('검증 ID가 분할 기록과 다릅니다.')
    for r in rows:
        if r['group_id']!=expected[r['id']] or any(r.get(k)!=v for k,v in original[r['id']].items()):
            raise ValueError('검증 서열/그룹이 원본과 다릅니다: '+r['id'])
    return rows

def evaluate(args,adapter=None):
    import torch
    rows=verified_validation_rows(args.data,args.split)
    set_seed(args.seed)
    model,tokenizer=load_model(adapter)
    total_nll=0.;total_tokens=0;per_sequence=[]
    with torch.inference_mode():
        for row in rows:
            encoded=encode_example(row,tokenizer)
            if len(encoded['input_ids'])>model.config.n_positions: raise ValueError('문맥 길이 초과')
            batch={k:torch.tensor([v],dtype=torch.long,device='cuda') for k,v in encoded.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
                labels=batch['labels'][:,1:].contiguous().view(-1)
                nll=torch.nn.functional.cross_entropy(logits[:,:-1].contiguous().float().view(-1,logits.shape[-1]),
                                                      labels,ignore_index=-100,reduction='sum')
            count=int((labels!=-100).sum().item());value=nll.item()
            if not count or not math.isfinite(value): raise RuntimeError('잘못된 NLL')
            total_nll+=value;total_tokens+=count
            per_sequence.append(dict(id=row['id'],group_id=row['group_id'],tokens=count,nll=value/count))
    if not total_tokens: raise ValueError('검증셋이 비었습니다.')
    metrics=dict(nll=total_nll/total_tokens,perplexity=math.exp(total_nll/total_tokens),tokens=total_tokens)
    output=args.output or new_output('LoRA_NLL' if adapter is not None else 'MAGE_NLL')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'evaluation.json',dict(metrics=metrics,per_sequence=per_sequence,seed=args.seed,
               model='LoRA' if adapter is not None else 'MAGE',split=str(args.split),
               adapter=str(adapter) if adapter is not None else None))
    print(json.dumps(metrics,indent=2));print('저장: '+str(output))

# ── 3. 데이터 선별 확인·유사 항체 그룹 분할·학습 토큰 생성 ─────────
def validate_rows(rows):
    ids = set()
    for r in rows:
        if not r.get('id') or r['id'] in ids:
            raise ValueError('Missing or duplicate id')
        ids.add(r['id'])
        for k in ('antigen_seq', 'VH_AA', 'VL_AA'):
            if not r.get(k) or not set(r[k]) <= AA:
                raise ValueError(f'{r["id"]}: invalid amino acid sequence in {k}')
        expected = r['antigen_seq']+'[SEP]'+r['VH_AA']+'[LC]'+r['VL_AA']
        if r.get('SEQ') != expected:
            raise ValueError(f'{r["id"]}: SEQ does not match antigen/VH/VL')
    if len(rows) < 2:
        raise ValueError('At least two rows required')

def chain_similarity(a, b):
    from rapidfuzz.distance import Levenshtein
    return 1.0 - Levenshtein.distance(a, b)/max(len(a), len(b))

def build_split(rows, threshold=0.90, seed=42, val_fraction=0.20):
    validate_rows(rows)
    if not 0 < threshold <= 1 or not 0 < val_fraction < 1:
        raise ValueError('Invalid similarity threshold or validation fraction')
    rows = sorted(rows, key=lambda r: r['id'])
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs, edges = [], []
    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            h = chain_similarity(a['VH_AA'], b['VH_AA'])
            l = chain_similarity(a['VL_AA'], b['VL_AA'])
            pairs.append((i, j, h, l))
            # Small tolerance handles e.g. a literal 90% boundary exactly.
            if max(h, l) >= threshold - 1e-12:
                parent[root(i)] = root(j)
                edges.append(dict(id_a=a['id'], id_b=b['id'], vh_similarity=h, vl_similarity=l))
    components = {}
    for i in range(len(rows)):
        components.setdefault(root(i), []).append(i)
    groups = sorted(components.values(), key=lambda g: rows[g[0]]['id'])
    if len(groups) < 2:
        raise ValueError('Only one connected group: a leakage-free split is impossible')

    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    # Subset-sum over group sizes. Seed controls equal-size alternative subsets.
    possible = {0: ()}
    for g in order:
        for count, subset in list(possible.items()):
            possible.setdefault(count+len(groups[g]), subset+(g,))
    target = max(1, min(len(rows)-1, int(len(rows)*val_fraction+0.5)))
    n_val = min((n for n in possible if 0 < n < len(rows)),
                key=lambda n: (abs(n-target), n))
    val_groups = set(possible[n_val])
    assignments, membership = [], {}
    for g, members in enumerate(groups):
        group_id = rows[members[0]]['id']
        split = 'validation' if g in val_groups else 'train'
        for i in members:
            membership[i] = split
            assignments.append(dict(id=rows[i]['id'], group_id=group_id, split=split))
    assignments.sort(key=lambda r: r['id'])
    crossing = [(i, j, h, l) for i, j, h, l in pairs if membership[i] != membership[j]]
    best_h = max(crossing, key=lambda p: p[2])
    best_l = max(crossing, key=lambda p: p[3])
    audit = dict(cross_partition_pairs=len(crossing),
                 threshold_violations=sum(max(h, l) >= threshold-1e-12 for _, _, h, l in crossing),
                 max_vh_similarity=best_h[2], max_vl_similarity=best_l[3],
                 closest_vh_ids=[rows[i]['id'] for i in best_h[:2]],
                 closest_vl_ids=[rows[i]['id'] for i in best_l[:2]])
    if audit['threshold_violations']:
        raise RuntimeError('Cross-partition similarity leakage')
    return dict(split_seed=seed, similarity_threshold=threshold,
                similarity_metric='1 - Levenshtein_distance / max(chain_lengths)',
                edge_rule='VH >= threshold OR VL >= threshold; transitive connected components',
                val_fraction_requested=val_fraction, validation_target=target,
                counts=dict(total=len(rows), train=len(rows)-n_val, validation=n_val, groups=len(groups)),
                group_sizes=sorted(map(len, groups), reverse=True),
                assignments=assignments, similarity_edges=edges, audit=audit)

def encode_example(row, tokenizer, max_length=None):
    ids = tokenizer.encode(row['SEQ'], add_special_tokens=False)
    sep = tokenizer.encode('[SEP]', add_special_tokens=False)
    lc = tokenizer.encode('[LC]', add_special_tokens=False)
    if len(sep) != 1 or len(lc) != 1 or ids.count(sep[0]) != 1 or ids.count(lc[0]) != 1:
        raise ValueError('Unexpected MAGE special-token encoding')
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError('EOS and PAD tokens must be defined')
    # Explicit EOS gives an antibody-completion target; never supervise padding.
    ids = ids + [tokenizer.eos_token_id]
    labels = ids.copy()
    boundary = ids.index(sep[0])+1
    labels[:boundary] = [-100]*boundary
    mask = [1]*len(ids)
    if max_length is not None:
        if len(ids) > max_length:
            raise ValueError('Refusing to truncate antibody example')
        padding = max_length-len(ids)
        ids += [tokenizer.pad_token_id]*padding
        labels += [-100]*padding
        mask += [0]*padding
    return dict(input_ids=ids, attention_mask=mask, labels=labels)

def prepare_split(data, output, seed=42, threshold=0.90, val_fraction=0.20):
    with Path(data).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    result = build_split(rows, threshold, seed, val_fraction)
    result.update(input_sha256=sha256(data), input_filename=Path(data).name,
                  rapidfuzz_version=importlib.metadata.version('rapidfuzz'),
                  script_sha256=sha256(__file__))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'split_manifest.json', result)
    by_id = {r['id']: r for r in rows}
    for split in ('train', 'validation'):
        chosen = [{**by_id[a['id']], 'group_id': a['group_id'], 'split': split}
                  for a in result['assignments'] if a['split'] == split]
        with (output/f'{split}.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(chosen[0]))
            writer.writeheader(); writer.writerows(chosen)
    with (output/'assignments.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'group_id', 'split'])
        writer.writeheader(); writer.writerows(result['assignments'])
    return result

# ── 4. LoRA 학습: 누적 gradient·검증·조기 종료·최적 모델 저장 ──────
class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.001):
        self.patience, self.min_delta = patience, min_delta
        self.best = self.significant_best = float('inf')
        self.bad_epochs = 0

    def observe(self, value):
        if not math.isfinite(value):
            raise ValueError('Non-finite validation loss')
        improved = value < self.best
        self.best = min(self.best, value)
        if value < self.significant_best-self.min_delta:
            self.significant_best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience

def accumulation_windows(order, size):
    if size < 1:
        raise ValueError('Accumulation size must be positive')
    for offset in range(0, len(order), size):
        yield order[offset:offset+size]

def antibody_loss_sum(logits, labels):
    import torch
    targets = labels[:, 1:].contiguous().view(-1)
    count = int((targets != -100).sum().item())
    if not count:
        raise ValueError('No supervised antibody tokens')
    nll = torch.nn.functional.cross_entropy(
        logits[:, :-1].contiguous().float().view(-1, logits.shape[-1]),
        targets, ignore_index=-100, reduction='sum')
    return nll, count

def load_verified_split(data, split):
    manifest = json.loads((split/'split_manifest.json').read_text(encoding='utf-8'))
    if sha256(data) != manifest['input_sha256']:
        raise ValueError('Input CSV hash differs from recorded split')
    with data.open(encoding='utf-8-sig', newline='') as handle:
        raw = list(csv.DictReader(handle))
    # 소스 파일 통합과 무관하게 분할 결과/누수 감사/행 내용을 직접 재검증한다.
    rebuilt = build_split(raw, manifest['similarity_threshold'], manifest['split_seed'], manifest['val_fraction_requested'])
    for key in ('assignments', 'counts', 'audit'):
        if rebuilt[key] != manifest[key]:
            raise ValueError('Split manifest mismatch: '+key)
    original = {r['id']: r for r in raw}
    datasets = {}
    for name in ('train', 'validation'):
        with (split/f'{name}.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        expected = sorted(a['id'] for a in manifest['assignments'] if a['split'] == name)
        if sorted(r['id'] for r in rows) != expected:
            raise ValueError(name+' IDs differ from manifest')
        for r in rows:
            if any(r.get(k) != v for k, v in original[r['id']].items()):
                raise ValueError(name+' row differs from original: '+r['id'])
        datasets[name] = sorted(rows, key=lambda r:r['id'])
    return manifest, datasets

def train_model(args):
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import PreTrainedTokenizerFast
    sys.path.insert(0, str(SOURCE))
    from progen.modeling_progen import ProGenForCausalLM

    output = args.output
    start = time.perf_counter()
    random.seed(args.train_seed); np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed); torch.cuda.manual_seed_all(args.train_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required')
    manifest, datasets = load_verified_split(args.data, args.split)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(CHECKPOINT), local_files_only=True)
    encoded = {name:[dict(id=r['id'], group_id=r['group_id'],
                         **encode_example(r, tokenizer)) for r in rows]
               for name, rows in datasets.items()}
    settings = dict(split_seed=manifest['split_seed'], train_seed=args.train_seed,
                    train_count=len(encoded['train']), validation_count=len(encoded['validation']),
                    r=TRAIN_CONFIG["rank"], alpha=TRAIN_CONFIG["alpha"], dropout=TRAIN_CONFIG["dropout"], targets=['qkv_proj', 'out_proj'],
                    base_dtype='bfloat16', adapter_dtype='float32', autocast='bfloat16',
                    microbatch_size=1, gradient_accumulation_steps=TRAIN_CONFIG["accumulation"],
                    loss_normalization='sum supervised token NLL / actual tokens in accumulation window',
                    supervision='VH+[LC]+VL+EOS; antigen, SEP and padding ignored',
                    optimizer='AdamW', learning_rate=TRAIN_CONFIG["learning_rate"], weight_decay=TRAIN_CONFIG["weight_decay"],
                    gradient_clip_norm=TRAIN_CONFIG["max_grad_norm"], scheduler='constant; no warmup',
                    max_epochs=args.epochs, patience=TRAIN_CONFIG["patience"], min_delta=TRAIN_CONFIG["min_delta"],
                    selection='raw minimum validation token NLL after a trained epoch',
                    early_stopping=f"{TRAIN_CONFIG['patience']} epochs without improvement > {TRAIN_CONFIG['min_delta']} from significant best trained epoch",
                    deterministic_algorithms=True, tf32=False, gradient_checkpointing=False,
                    use_cache=False, cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                    epoch_shuffle='random.Random(train_seed + epoch), canonical sorted input IDs',
                    fresh_from_original_MAGE=True, validation_is_not_test=True,
                    command=[sys.executable, '-X', 'utf8', *sys.argv])
    write_json(output/'settings.json', settings)
    write_json(output/'split_manifest.json', manifest)
    orders = {}
    for epoch in range(1, args.epochs+1):
        order = list(range(len(encoded['train'])))
        random.Random(args.train_seed+epoch).shuffle(order)
        orders[str(epoch)] = order
    write_json(output/'epoch_orders.json', {e:[encoded['train'][i]['id'] for i in order] for e,order in orders.items()})
    environment = dict(python=sys.version, packages={p:importlib.metadata.version(p) for p in
                       ['torch','transformers','peft','rapidfuzz','accelerate','numpy','tokenizers','safetensors']},
                       cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(0), hash_seed=os.environ.get('PYTHONHASHSEED'),
                       nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),
                       progen_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip())
    write_json(output/'environment.json', environment)
    (output/'requirements-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    hash_files = dict(input_csv=args.data, split_manifest=args.split/'split_manifest.json',
                      train_csv=args.split/'train.csv', validation_csv=args.split/'validation.csv',
                      model_weights=CHECKPOINT/'pytorch_model.bin', model_config=CHECKPOINT/'config.json',
                      tokenizer=CHECKPOINT/'tokenizer.json', training_source=Path(__file__),
                      data_source=Path(__file__), progen_source=SOURCE/'progen/modeling_progen.py')
    write_json(output/'input_hashes.json', {k:sha256(p) for k,p in hash_files.items()})

    print('Loading original MAGE and initializing fresh LoRA...', flush=True)
    model, loading = ProGenForCausalLM.from_pretrained(str(CHECKPOINT), torch_dtype=torch.bfloat16,
                     low_cpu_mem_usage=True, local_files_only=True, output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('Model/checkpoint mismatch: '+str(loading))
    if max(len(r['input_ids']) for rows in encoded.values() for r in rows) > model.config.n_positions:
        raise ValueError('Input exceeds model context; no truncation allowed')
    model.config.use_cache=False
    model.requires_grad_(False)
    targets = [n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)
               and n.startswith('transformer.h.') and n.endswith(('.attn.qkv_proj','.attn.out_proj'))]
    if len(targets) != model.config.n_layer*2:
        raise RuntimeError('Unexpected LoRA targets')
    model = get_peft_model(model,LoraConfig(r=TRAIN_CONFIG["rank"],lora_alpha=TRAIN_CONFIG["alpha"],lora_dropout=TRAIN_CONFIG["dropout"],target_modules=targets,bias='none'))
    for n,p in model.named_parameters():
        if p.requires_grad:
            if 'lora_' not in n: raise RuntimeError('Non-LoRA parameter trainable')
            p.data=p.data.float()
    model.to('cuda')
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    frozen_versions = [p._version for p in frozen]
    optimizer = torch.optim.AdamW(trainable,lr=TRAIN_CONFIG["learning_rate"],weight_decay=TRAIN_CONFIG["weight_decay"],foreach=False)
    torch.cuda.reset_peak_memory_stats()
    tokenizer.save_pretrained(output/'토크나이저')

    def forward(row):
        batch = {k:torch.tensor([row[k]],dtype=torch.long,device='cuda')
                 for k in ('input_ids','attention_mask','labels')}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits = model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
            nll,count = antibody_loss_sum(logits,batch['labels'])
        if not torch.isfinite(nll): raise RuntimeError('Non-finite loss')
        return nll,count

    def evaluate():
        model.eval()
        total_nll=0.; total_tokens=0; per_sequence=[]
        with torch.inference_mode():
            for row in encoded['validation']:
                nll,count=forward(row)
                value=nll.item()
                total_nll+=value; total_tokens+=count
                per_sequence.append(dict(id=row['id'],group_id=row['group_id'],
                                         tokens=count,nll=value/count))
        value=total_nll/total_tokens
        return dict(nll=value,perplexity=math.exp(value),tokens=total_tokens),per_sequence

    baseline, baseline_rows=evaluate()
    write_json(output/'baseline_validation.json',dict(metrics=baseline,per_sequence=baseline_rows))
    print('Baseline validation: '+json.dumps(baseline),flush=True)
    monitor=EarlyStopping(patience=TRAIN_CONFIG["patience"],min_delta=TRAIN_CONFIG["min_delta"])
    history=[]; step=0; best_epoch=None; stopped=False
    for epoch in range(1,args.epochs+1):
        epoch_start=time.perf_counter(); model.train()
        train_nll=0.; train_tokens=0
        for window in accumulation_windows(orders[str(epoch)],TRAIN_CONFIG['accumulation']):
            optimizer.zero_grad(set_to_none=True)
            window_tokens=sum(sum(v != -100 for v in encoded['train'][i]['labels'][1:]) for i in window)
            for i in window:
                nll,count=forward(encoded['train'][i])
                train_nll+=nll.item(); train_tokens+=count
                (nll/window_tokens).backward()
                del nll
            if step == 0:
                if any(p.grad is None for p in trainable) or not any(torch.count_nonzero(p.grad).item() for p in trainable):
                    raise RuntimeError('Missing/nonzero adapter gradient check failed')
            grad_norm=torch.nn.utils.clip_grad_norm_(trainable,TRAIN_CONFIG["max_grad_norm"],error_if_nonfinite=True)
            optimizer.step(); step+=1
            if step % 5 == 0:
                progress=dict(status='running',epoch=epoch,optimizer_step=step,
                              completed_supervised_tokens=train_tokens,elapsed_seconds=time.perf_counter()-start)
                write_json(output/'progress.json',progress)
                print(f'Epoch {epoch}: optimizer step {step}',flush=True)
        optimizer.zero_grad(set_to_none=True)
        validation,_=evaluate()
        improved,stopped=monitor.observe(validation['nll'])
        if improved:
            best_epoch=epoch
            model.save_pretrained(output/'최적어댑터',safe_serialization=True)
        model.save_pretrained(output/'마지막어댑터',safe_serialization=True)
        checkpoint=dict(epoch=epoch,optimizer_step=step,optimizer=optimizer.state_dict(),
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                        numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                        early_stopping=dict(best=monitor.best,significant_best=monitor.significant_best,
                                            bad_epochs=monitor.bad_epochs),best_epoch=best_epoch)
        torch.save(checkpoint,output/'last_training_state.pt')
        if any(p.grad is not None or p._version != v for p,v in zip(frozen,frozen_versions)):
            raise RuntimeError('Frozen base invariant failed')
        record=dict(epoch=epoch,optimizer_step=step,train_nll=train_nll/train_tokens,
                    train_tokens=train_tokens,validation_nll=validation['nll'],
                    validation_perplexity=validation['perplexity'],best_epoch=best_epoch,
                    patience_counter=monitor.bad_epochs,seconds=time.perf_counter()-epoch_start)
        history.append(record)
        write_json(output/'history.json',history)
        with (output/'history.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(record));writer.writeheader();writer.writerows(history)
        print(json.dumps(record),flush=True)
        write_json(output/'progress.json',dict(status='running',**record))
        if stopped: break

    # Load the selected checkpoint rather than reporting the last epoch as best.
    state=load_file(str(output/'최적어댑터/adapter_model.safetensors'))
    set_peft_model_state_dict(model,state)
    restored=get_peft_model_state_dict(model)
    if set(state) != set(restored) or not all(torch.equal(v,restored[k].detach().cpu()) for k,v in state.items()):
        raise RuntimeError('Best adapter reload failed')
    best_metrics,best_rows=evaluate()
    if abs(best_metrics['nll']-monitor.best) > 1e-6:
        raise RuntimeError('Best checkpoint validation differs after reload')
    write_json(output/'best_validation_per_sequence.json',dict(metrics=best_metrics,per_sequence=best_rows))
    result=dict(status='completed',finished_utc=datetime.now(timezone.utc).isoformat(),
                epochs_completed=len(history),optimizer_steps=step,early_stopped=stopped,best_epoch=best_epoch,
                baseline_validation=baseline,best_validation=best_metrics,
                validation_nll_change=best_metrics['nll']-baseline['nll'],
                improved_over_baseline=best_metrics['nll'] < baseline['nll'],
                frozen_base_unchanged=True,best_adapter_reload_verified=True,
                trainable_parameters=sum(p.numel() for p in trainable),target_layer_count=len(targets),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                elapsed_seconds=time.perf_counter()-start,settings=settings,
                limitations='One seed; validation used for checkpoint selection; no independent test set or experimental binding validation')
    write_json(output/'result.json',result)
    write_json(output/'progress.json',dict(status='completed',epochs=len(history),best_epoch=best_epoch))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

# ── 5. 저장된 결과 시각화: GPU 로딩·학습 없음 ─────────────────────
def pair_records(baseline, adapted):
    def index(records):
        result = {}
        for r in records:
            if r['id'] in result or not math.isfinite(r['nll']) or r['nll'] < 0 or r['tokens'] <= 0:
                raise ValueError('Duplicate ID or invalid NLL/token count')
            result[r['id']] = r
        return result
    before, after = index(baseline), index(adapted)
    if not before or set(before) != set(after):
        raise ValueError('Baseline and LoRA must contain exactly the same nonempty ID set')
    paired = []
    for number, name in enumerate(sorted(before), 1):
        b, a = before[name], after[name]
        if b['tokens'] != a['tokens'] or b['group_id'] != a['group_id']:
            raise ValueError('Target token count or group changed for '+name)
        paired.append(dict(alias=f'V{number:02d}', id=name, group_id=b['group_id'],
                           tokens=b['tokens'], baseline_nll=b['nll'], lora_nll=a['nll'],
                           delta_nll=a['nll']-b['nll'],
                           baseline_perplexity=math.exp(b['nll']), lora_perplexity=math.exp(a['nll'])))
    return paired

def summarize_pairs(rows):
    tokens = sum(r['tokens'] for r in rows)
    baseline = sum(r['tokens']*r['baseline_nll'] for r in rows)/tokens
    lora = sum(r['tokens']*r['lora_nll'] for r in rows)/tokens
    return dict(pairs=len(rows), groups=len({r['group_id'] for r in rows}), tokens=tokens,
                baseline_nll=baseline, lora_nll=lora,
                baseline_perplexity=math.exp(baseline), lora_perplexity=math.exp(lora),
                improved_pairs=sum(r['delta_nll'] < -1e-12 for r in rows),
                worsened_pairs=sum(r['delta_nll'] > 1e-12 for r in rows),
                unchanged_pairs=sum(abs(r['delta_nll']) <= 1e-12 for r in rows),
                delta_nll=lora-baseline, nll_reduction_percent=100*(baseline-lora)/baseline if baseline else None)

def plot_results(args):
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = args.run_dir
    read = lambda name: json.loads((run/name).read_text(encoding='utf-8'))
    baseline = read('baseline_validation.json')
    adapted = read('best_validation_per_sequence.json')
    result = read('result.json')
    if result['status'] != 'completed':
        raise ValueError('Training run is not complete')
    rows = pair_records(baseline['per_sequence'], adapted['per_sequence'])
    summary = summarize_pairs(rows)
    for saved, key in ((baseline, 'baseline'), (adapted, 'lora')):
        if saved['metrics']['tokens'] != summary['tokens'] or not math.isclose(
                saved['metrics']['nll'], summary[key+'_nll'], rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError('Saved aggregate disagrees with per-sequence metrics')
    if not math.isclose(result['best_validation']['nll'], summary['lora_nll'], abs_tol=1e-9):
        raise ValueError('Selected checkpoint metrics disagree')
    with (run/'history.csv').open(encoding='utf-8', newline='') as handle:
        history = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(handle)]
    best = result['best_epoch']
    selected = [h for h in history if int(h['epoch']) == best]
    if len(selected) != 1 or not math.isclose(selected[0]['validation_nll'], summary['lora_nll'], abs_tol=1e-9):
        raise ValueError('Best epoch disagrees with history')
    output = args.output_dir or run/'시각화'
    output.mkdir(parents=True, exist_ok=True)
    with (output/'paired_nll.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary.update(best_epoch=best, train_seed=result['settings']['train_seed'],
                   matplotlib_version=importlib.metadata.version('matplotlib'),
                   source_files=['baseline_validation.json','best_validation_per_sequence.json','history.csv','result.json'],
                   interpretation='Descriptive paired validation comparison; checkpoint selected on same validation set; no independent test, p-value or binding claim')
    (output/'plot_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','svg.fonttype':'none','pdf.fonttype':42})
    blue, teal, red, grey = '#3264A0', '#14806C', '#B95050', '#75808B'
    fig, axes = plt.subplots(1,3,figsize=(16.5,6.5),gridspec_kw={'width_ratios':[1.2,1.,1.15]})
    fig.subplots_adjust(left=0.065,right=0.985,bottom=0.19,top=0.79,wspace=0.37)
    fig.suptitle('PD-L1 antibody LoRA: validation likelihood evaluation',x=0.065,y=0.965,
                 ha='left',fontsize=19,fontweight='bold')
    fig.text(0.065,0.905,f"Same {summary['pairs']} validation pairs | {summary['groups']} sequence groups | "
             f"{summary['tokens']:,} scored tokens | seed {summary['train_seed']} | selected epoch {best}",
             fontsize=11,color='#46515C')

    ax=axes[0]
    epochs=[h['epoch'] for h in history]
    ax.plot(epochs,[h['train_nll'] for h in history],'o-',color=grey,label='Online training NLL',markersize=4)
    ax.plot([0]+epochs,[summary['baseline_nll']]+[h['validation_nll'] for h in history],
            'o-',color=teal,label='Validation NLL',markersize=4)
    ax.axhline(summary['baseline_nll'],color=blue,ls='--',lw=1.2,label='Original MAGE baseline')
    ax.axvline(best,color=grey,ls=':',lw=1)
    ax.scatter([best],[summary['lora_nll']],s=95,facecolors='white',edgecolors=teal,lw=2,zorder=5)
    ax.annotate(f"Best: epoch {best}\nNLL {summary['lora_nll']:.4f}",
                (best,summary['lora_nll']),xytext=(12,25),textcoords='offset points',
                fontsize=9,arrowprops={'arrowstyle':'-','color':grey})
    ax.set(title='A  Training dynamics',xlabel='Epoch (0 = before training)',ylabel='Token-normalized NLL (nats/token)')
    ax.set_xticks(range(0,int(max(epochs))+1))
    ax.legend(frameon=False,fontsize=8,loc='lower left')
    ax.grid(axis='y',color='#E5E8EB',lw=0.7)

    ax=axes[1]
    data=[[r[k] for r in rows] for k in ('baseline_nll','lora_nll')]
    ax.boxplot(data,positions=[0,1],widths=0.32,showfliers=False,patch_artist=True,
               boxprops={'facecolor':'#EDF0F3','edgecolor':'#87929C'},
               medianprops={'color':'#46515C'},whiskerprops={'color':'#87929C'},capprops={'color':'#87929C'})
    for r in rows:
        color=teal if r['delta_nll'] < 0 else red
        ax.plot([0,1],[r['baseline_nll'],r['lora_nll']],color=color,alpha=0.45,lw=0.8)
        ax.scatter([0,1],[r['baseline_nll'],r['lora_nll']],color=[blue,teal],s=15,zorder=3)
    ax.scatter([0,1],[summary['baseline_nll'],summary['lora_nll']],marker='D',s=58,color='#1E2933',zorder=5,
               label='Token-weighted mean')
    ax.set_xticks([0,1],['Original MAGE',f'LoRA epoch {best}'])
    ax.set(title='B  Matched antibody pairs',ylabel='Per-pair NLL (nats/token)',xlabel='Model (each line = same VH/VL pair)')
    ax.set_xlim(-0.35,1.35)
    ax.legend(frameon=False,fontsize=8,loc='upper right')
    ax.grid(axis='y',color='#E5E8EB',lw=0.7)

    ax=axes[2]
    ranked=sorted(rows,key=lambda r:(r['delta_nll'],r['id']))
    ax.barh(range(len(ranked)),[r['delta_nll'] for r in ranked],height=0.68,
            color=[teal if r['delta_nll'] < 0 else red for r in ranked])
    ax.axvline(0,color=grey,lw=1)
    ax.set_yticks(range(len(ranked)),[r['alias'] for r in ranked],fontsize=8)
    ax.invert_yaxis()
    ax.set(title='C  Change for every validation pair',xlabel='Delta NLL = LoRA - original (nats/token)',
           ylabel='Validation pair ID (see paired_nll.csv)')
    ax.text(0.03,0.02,f"{summary['improved_pairs']}/{summary['pairs']} pairs improved\nNegative = lower NLL",transform=ax.transAxes,
            fontsize=9,va='bottom',bbox={'facecolor':'white','edgecolor':'none','alpha':0.9})
    ax.grid(axis='x',color='#E5E8EB',lw=0.7);ax.set_axisbelow(True)
    fig.text(0.065,0.09,f"Token-weighted validation: NLL {summary['baseline_nll']:.4f} -> {summary['lora_nll']:.4f} | "
             f"PPL {summary['baseline_perplexity']:.4f} -> {summary['lora_perplexity']:.4f}",fontsize=10)
    fig.text(0.065,0.047,'Scored: VH + [LC] + VL + EOS, conditioned on antigen. Validation used for checkpoint selection; one training run.',
             fontsize=8,color='#59636C')
    fig.text(0.065,0.018,'Online training uses dropout and changing weights; validation uses eval mode. No uncertainty bands or binding-affinity claim.',
             fontsize=8,color='#59636C')
    for extension in ('png','svg'):
        fig.savefig(output/f'lora_validation_comparison.{extension}',dpi=300,facecolor='white')
    print(json.dumps(summary,indent=2))
    print('Figures and paired CSV saved to: '+str(output.resolve()))
    if args.show:
        plt.show()
    else:
        plt.close(fig)


def rejection_reasons(parsed):
    reasons=[]
    if not parsed['termination_found']: reasons.append('missing_termination')
    if parsed['lc_token_count']!=1: reasons.append('invalid_lc_count')
    if not parsed['standard_amino_acids_only']: reasons.append('nonstandard_or_missing_chain')
    if parsed['vh_length'] is not None and parsed['vh_length']<100: reasons.append('short_vh')
    if parsed['vl_length'] is not None and parsed['vl_length']<100: reasons.append('short_vl')
    return reasons

def enable_generation_cache(model):
    """ProGen의 옛 past 인자를 Transformers의 past_key_values와 연결한다."""
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    original=base.prepare_inputs_for_generation
    def prepare(input_ids,past=None,**kwargs):
        incoming=kwargs.pop('past_key_values',None)
        return original(input_ids,past=past if past is not None else incoming,**kwargs)
    base.prepare_inputs_for_generation=prepare

def acquire_generation_lock(output,filename='.generation.lock'):
    """프로세스가 끝나면 OS가 자동 해제하는 출력 폴더 쓰기 잠금."""
    output.mkdir(parents=True,exist_ok=True)
    handle=(output/filename).open('a+b')
    if handle.seek(0,2)==0: handle.write(b'0');handle.flush()
    handle.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close();return None
    return handle


def generate_many(args,adapter=None):
    args.output=args.output or new_output('LoRA_대량생성' if adapter else 'MAGE_대량생성')
    lock=acquire_generation_lock(args.output)
    if lock is None:
        print(json.dumps(dict(status='already_running',output=str(args.output))),flush=True);return
    try: _generate_many(args,adapter)
    finally: lock.close()


def _generate_many(args,adapter=None):
    """모델을 한 번만 읽고 생성 시도 전부를 배치별로 원자적으로 보존한다."""
    import torch
    if args.count<1 or args.batch_size<1: raise ValueError('count/batch_size must be positive')
    antigen=args.antigen or read_antigen(args.data)
    label='LoRA' if adapter is not None else 'MAGE'
    output=args.output or new_output(label+'_bulk')
    output.mkdir(parents=True,exist_ok=True)
    batches=output/'원본배치';batches.mkdir(exist_ok=True)
    settings=dict(model=label,count=args.count,batch_size=args.batch_size,seed=args.seed,
                  batch_seed_rule='seed + zero-based batch start index',top_p=args.top_p,
                  temperature=args.temperature,max_new_tokens=args.max_new_tokens,antigen=antigen,
                  dtype='bfloat16',use_cache=True,cache_api='past_key_values mapped to ProGen past',
                  data_sha256=sha256(args.data),checkpoint_sha256=sha256(CHECKPOINT/'pytorch_model.bin'),
                  progen_source_sha256=sha256(SOURCE/'progen/modeling_progen.py'),
                  tokenizer_sha256=sha256(CHECKPOINT/'tokenizer.json'),
                  adapter_sha256=sha256(adapter/'adapter_model.safetensors') if adapter is not None else None)
    manifest=output/'generation_settings.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8'))!=settings:
            raise ValueError('Resume settings differ from existing generation settings')
    else: write_json(manifest,settings)
    runtime=dict(python=sys.version,packages={n:importlib.metadata.version(n) for n in
        ('torch','transformers','tokenizers','peft','numpy')},code_sha256=sha256(__file__),
        source_recorded_at='process launch',seed=args.seed,torch_deterministic=True,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    runtime_path=output/'runtime_provenance.json'
    if runtime_path.exists(): runtime_path=output/('runtime_resume_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    write_json(runtime_path,runtime)
    set_seed(args.seed)
    model,tokenizer=load_model(adapter);enable_generation_cache(model)
    prompt=tokenizer(antigen+'[SEP]',return_tensors='pt')
    if prompt['input_ids'].shape[1]+args.max_new_tokens>model.config.n_positions: raise ValueError('Context too long')
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for offset in range(0,args.count,args.batch_size):
        size=min(args.batch_size,args.count-offset)
        path=batches/f'{offset:06d}.json'
        if path.exists():
            saved=json.loads(path.read_text(encoding='utf-8'))
            if [r['index'] for r in saved]!=list(range(offset,offset+size)):
                raise ValueError('Invalid saved batch: '+str(path))
            continue
        set_seed(args.seed+offset)
        inputs={k:v.repeat(size,1).to('cuda') for k,v in prompt.items() if k in ('input_ids','attention_mask')}
        length=inputs['input_ids'].shape[1];batch_start=time.perf_counter()
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=True,
                top_p=args.top_p,temperature=args.temperature,use_cache=True,
                pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        records=[]
        for i,new in enumerate(generated[:,length:].tolist()):
            parsed=parse_antibody(new,tokenizer)
            records.append(dict(id=f'{label}_{offset+i+1:04d}',model=label,index=offset+i,
                                batch_seed=args.seed+offset,raw_ids=new,
                                raw_text=tokenizer.decode(new),**parsed,
                                rejection_reasons=rejection_reasons(parsed)))
        temporary=path.with_suffix('.tmp');write_json(temporary,records);temporary.replace(path)
        progress=dict(status='generating',model=label,completed=offset+size,total=args.count,
                      last_batch_seconds=time.perf_counter()-batch_start,
                      elapsed_seconds=time.perf_counter()-start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    records=[]
    for offset in range(0,args.count,args.batch_size):
        records.extend(json.loads((batches/f'{offset:06d}.json').read_text(encoding='utf-8')))
    if len(records)!=args.count: raise RuntimeError('Generation count mismatch')
    write_json(output/'raw_candidates.json',records)
    counts=dict(status='generated',model=label,total=len(records),
                basic_pass=sum(not r['rejection_reasons'] for r in records),
                unique_pairs=len({(r['vh'],r['vl']) for r in records if not r['rejection_reasons']}),
                elapsed_seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(output/'progress.json',counts);print(json.dumps(counts),flush=True)

def unique_selection(rows,count):
    chosen=[];seen=set()
    for row in rows:
        pair=(row['VH_AA'],row['VL_AA'])
        if pair in seen: continue
        chosen.append(row);seen.add(pair)
        if len(chosen)>=count: break
    return chosen


# ── 6. 논문 방식 생성물 분석: Linux의 ANARCI/BioPhi만 지연 import ────
REGIONS = ('FWR1','CDR1','FWR2','CDR2','FWR3','CDR3','FWR4')
ANALYSIS_PYTHON = '/root/.cache/pdl1-analysis/env/bin/python'


def analysis_package_version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        # ANARCI의 bioconda 패키지는 Python distribution metadata가 없다.
        records=sorted((Path(sys.prefix)/'conda-meta').glob(name+'-*.json'))
        if len(records)!=1: raise
        return json.loads(records[0].read_text(encoding='utf-8'))['version']


def select_paper_candidates(rows):
    """RBD의 10+5+5 규칙. 중복은 보충하지 않고 부족 수를 공개한다."""
    from collections import Counter
    pool=unique_selection([r for r in rows if r.get('humanness_pass')
          and .85<=r['v_identity.H']<1
          and r['max_identity_original_CDR3.H']<1
          and r['max_identity_train85_CDR3.H']<1],len(rows)+1)
    genes=sorted(Counter(r['v_gene.H'] for r in pool).items(),key=lambda x:(-x[1],x[0]))[:5]
    chosen=[];seen=set()
    def take(row,route):
        pair=(row['VH_AA'],row['VL_AA'])
        if pair not in seen:
            chosen.append(dict(row,selection_route=route));seen.add(pair)
    for gene,n in genes:
        ranked=sorted([r for r in pool if r['v_gene.H']==gene],key=lambda r:(r['v_identity.H'],r['id']))
        # 원본 노트북의 int(n*p)-1 (소수 표본의 -1도 포함)을 기록한다.
        for percentile in (.2,.8):
            take(ranked[int(n*percentile)-1],f'Vgene_{gene}_p{int(percentile*100)}')
    strata_count=len(chosen)
    remaining=[r for r in pool if (r['VH_AA'],r['VL_AA']) not in seen
               and not(r['max_identity_known106_CDR3.H']>=.9 and r['max_identity_known106_VH_AA']>=.9)]
    for field,route in [('max_identity_known106_CDR3.H','known_CDRH3'),('max_identity_known106_VH_AA','known_VH')]:
        ranked=sorted([r for r in remaining if (r['VH_AA'],r['VL_AA']) not in seen],
                      key=lambda r:(-r[field],r['id']))[:5]
        for row in ranked: take(row,route)
    return chosen,dict(target=20,selected=len(chosen),shortage=20-len(chosen),
                       eligible_unique_pairs=len(pool),stratified_selected=strata_count,
                       top_v_genes=genes,low_n_quantile_rule='int(n*p)-1; duplicate pair slots not filled')


def campaign_records(folder):
    complete=folder/'raw_candidates.json'
    if complete.exists(): return json.loads(complete.read_text(encoding='utf-8'))
    return [r for path in sorted((folder/'원본배치').glob('*.json'))
            for r in json.loads(path.read_text(encoding='utf-8'))]


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row)) or ['id','VH_AA','VL_AA','selection_route']
    with Path(path).open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def chain_from_annotation(annotation):
    from abnumber import Chain, Position
    aa_dict={Position(annotation['chain_type'],n,letter,'imgt'):aa
             for n,letter,aa in annotation['positions']}
    return Chain(sequence=None,scheme='imgt',cdr_definition='imgt',aa_dict=aa_dict,
                 chain_type=annotation['chain_type'],tail='',species='human',
                 v_gene=annotation['v_gene'],j_gene=annotation['j_gene'])


def annotate_sequences(sequences,cache):
    """한 고유 서열당 ANARCI를 한 번만 실행한다. 실패도 캐시한다."""
    from anarci import anarci
    cache.mkdir(parents=True,exist_ok=True)
    result={};pending=[]
    for sequence in sorted(set(sequences)):
        key=hashlib.sha256(sequence.encode()).hexdigest();path=cache/f'{key}.json'
        if path.exists(): result[sequence]=json.loads(path.read_text(encoding='utf-8'))
        else: pending.append((key,sequence))
    for offset in range(0,len(pending),128):
        batch=pending[offset:offset+128]
        numbers,details,_=anarci(batch,scheme='imgt',assign_germline=True,allowed_species=['human'],ncpu=8)
        for (key,sequence),domains,info in zip(batch,numbers,details):
            annotation=dict(input_sequence=sequence,success=False,reasons=[])
            if not domains or len(domains)!=1:
                annotation['reasons']=['no_domain' if not domains else 'multiple_domains']
            else:
                numbered,start,end=domains[0];d=info[0];g=d.get('germlines',{})
                annotation.update(chain_type=d['chain_type'],v_gene=g['v_gene'][0][1],
                    j_gene=g['j_gene'][0][1],v_identity=float(g['v_gene'][1]),
                    j_identity=float(g['j_gene'][1]),start=start,end=end,
                    positions=[[n,letter.strip(),aa] for (n,letter),aa in numbered if aa!='-'])
                chain=chain_from_annotation(annotation)
                annotation.update(sequence=chain.seq,regions={r:getattr(chain,r.lower().replace('fwr','fr')+'_seq') for r in REGIONS})
                if any(not s for s in annotation['regions'].values()): annotation['reasons'].append('missing_region')
                if len(chain.seq)<100: annotation['reasons'].append('short_variable_domain')
                annotation['success']=not annotation['reasons']
            write_json(cache/f'{key}.json',annotation);result[sequence]=annotation
        print(json.dumps(dict(phase='ANARCI',new_completed=min(offset+128,len(pending)),new_total=len(pending))),flush=True)
    return result


def paired_annotation(row,annotations):
    """원시 VH/VL을 보존하고 번호가 매겨진 가변 도메인을 별도로 저장한다."""
    out=dict(row);reasons=list(row.get('rejection_reasons',[]))
    if not reasons:
        for label,key,expected in [('H','vh',{'H'}),('L','vl',{'K','L'})]:
            a=annotations[row[key]]
            reasons.extend(label+':'+reason for reason in a['reasons'])
            if a.get('chain_type') not in expected: reasons.append(label+':wrong_chain_type')
            if a.get('success'):
                out.update({f'{region}.{label}':seq for region,seq in a['regions'].items()})
                out['V'+label+'_AA']=a['sequence']
                out['v_gene.'+label]=a['v_gene'].split('*')[0]
                out['v_allele.'+label]=a['v_gene'];out['v_identity.'+label]=a['v_identity']
                out['domain_start.'+label]=a['start'];out['domain_end.'+label]=a['end']
                out['non_domain_residues.'+label]=len(row[key])-len(a['sequence'])
    out['annotation_reasons']=reasons;out['annotation_pass']=not reasons
    return out


def add_reference_metrics(rows,references):
    import numpy as np
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    fields=['VH_AA','VL_AA']+[f'{r}.{c}' for c in ('H','L') for r in REGIONS]
    for refname,refrows in references.items():
        for field in fields:
            # 원본 데이터에서 주석이 없는 영역은 거리 계산에서 제외한다.
            choices=sorted({r.get(field,'') for r in refrows if r.get(field)})
            if not choices: raise ValueError(f'No {field} reference in {refname}')
            linked={}
            if field in ('VH_AA','VL_AA'):
                chain=field[1]
                for reference in refrows:
                    sequence=reference.get(field,'')
                    if sequence and (sequence not in linked or
                        sum(bool(reference.get(f'{r}.{chain}')) for r in REGIONS)>
                        sum(bool(linked[sequence].get(f'{r}.{chain}')) for r in REGIONS)):
                        linked[sequence]=reference
            queries=[r[field] for r in rows]
            for offset in range(0,len(rows),64):
                distance=process.cdist(queries[offset:offset+64],choices,scorer=Levenshtein.distance,
                                       dtype=np.uint16,workers=4)
                # 최소 편집거리와 최대 정규화 유사도를 각각 계산한다.
                identity=process.cdist(queries[offset:offset+64],choices,scorer=Levenshtein.normalized_similarity,
                                       dtype=np.float64,workers=4)
                for i,row in enumerate(rows[offset:offset+64]):
                    row[f'min_edit_{refname}_{field}']=int(distance[i].min())
                    row[f'max_identity_{refname}_{field}']=float(identity[i].max())
                    if field in ('VH_AA','VL_AA'):
                        nearest=linked[choices[int(distance[i].argmin())]]
                        row[f'nearest_chain_reference_{refname}_{field}']=nearest.get('id',nearest.get('Index Name',''))
                        for region in REGIONS:
                            regionfield=f'{region}.{chain}'
                            refseq=nearest.get(regionfield,'')
                            row[f'chain_match_edit_{refname}_{regionfield}']=Levenshtein.distance(row[regionfield],refseq) if refseq else None
        print(json.dumps(dict(phase='reference_distances',reference=refname,n=len(refrows))),flush=True)


def within_generated_metrics(rows):
    import numpy as np
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    summaries=[]
    for field in [f'CDR{i}.{c}' for c in ('H','L') for i in (1,2,3)]+['FR.H','FR.L']:
        values=[r[field] if field in r else ''.join(r[f'{f}.{field[-1]}'] for f in REGIONS if f.startswith('FWR')) for r in rows]
        if len(values)<2: summaries.append(dict(region=field,n_pairs=0,mean=None,std=None));continue
        matrix=process.cdist(values,values,scorer=Levenshtein.normalized_distance,dtype=np.float64,workers=4)
        distances=matrix[np.triu_indices(len(values),1)]
        summaries.append(dict(region=field,n_pairs=len(distances),mean=float(distances.mean()),std=float(distances.std())))
    return summaries


def score_oasis_pair(task):
    """공식 API를 독립 CPU 프로세스에서 호출한다. 서열/점수 근사는 하지 않는다."""
    from biophi.humanization.methods.humanness import OASisParams,get_antibody_humanness
    key,vh,vl,database=task
    h=get_antibody_humanness(chain_from_annotation(vh),chain_from_annotation(vl),
                            OASisParams(oasis_db_path=database,min_fraction_subjects=.10))
    score=dict(oasis_percentile=float(h.get_oasis_percentile(.10)),oasis_identity=float(h.get_oasis_identity(.10)),
               oasis_percentile_H=float(h.vh.get_oasis_percentile(.10)),oasis_percentile_L=float(h.vl.get_oasis_percentile(.10)))
    if not all(math.isfinite(v) for v in score.values()): raise ValueError('Non-finite OASis score: '+key)
    return key,score


def analysis_worker(args):
    output=args.campaign_dir/'분석'
    lock=acquire_generation_lock(output,filename='.analysis.lock')
    while lock is None:
        print(json.dumps(dict(phase='waiting_for_other_analysis',output=str(output))),flush=True)
        time.sleep(8);lock=acquire_generation_lock(output,filename='.analysis.lock')
    try: _analysis_worker(args)
    finally: lock.close()


def _analysis_worker(args):
    """WSL에서 수행. --annotation-only로 OASis 다운로드 전 주석을 완료할 수 있다."""
    import zipfile
    import io
    from collections import Counter
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    output=args.campaign_dir/'분석';output.mkdir(parents=True,exist_ok=True)
    load_verified_split(args.data,args.split)
    with args.data.open(encoding='utf-8-sig',newline='') as f: known=list(csv.DictReader(f))
    with (args.split/'train.csv').open(encoding='utf-8-sig',newline='') as f: train=list(csv.DictReader(f))
    raw=[r for model in ('MAGE','LoRA') for r in campaign_records(args.campaign_dir/model)]
    sequences=[r[c] for r in raw if not r['rejection_reasons'] for c in ('vh','vl')]
    sequences += [r[c] for r in known for c in ('VH_AA','VL_AA')]
    cache=output/'ANARCI_고유서열'
    annotation_settings=dict(scheme='imgt',cdr_definition='imgt',allowed_species=['human'],assign_germline=True,
                             versions={n:analysis_package_version(n) for n in ('abnumber','anarci')})
    annotation_manifest=output/'annotation_settings.json'
    if annotation_manifest.exists() and json.loads(annotation_manifest.read_text(encoding='utf-8'))!=annotation_settings:
        raise ValueError('Annotation settings differ from cached annotations')
    write_json(annotation_manifest,annotation_settings)
    annotations=annotate_sequences(sequences,cache)
    annotated=[paired_annotation(r,annotations) for r in raw]
    known_annotated=[paired_annotation(dict(id=r['id'],vh=r['VH_AA'],vl=r['VL_AA']),annotations) for r in known]
    # 실패한 알려진 항체를 임의 영역으로 채우지 않는다.
    known_valid=[r for r in known_annotated if r['annotation_pass']]
    trainids={r['id'] for r in train};train_valid=[r for r in known_valid if r['id'] in trainids]
    write_json(output/'generated_annotated.json',annotated)
    write_json(output/'known106_annotations.json',known_annotated)
    if args.annotation_only:
        print(json.dumps(dict(phase='annotation_ready',attempts=len(raw),annotated=sum(r['annotation_pass'] for r in annotated))),flush=True);return
    for model in ('MAGE','LoRA'):
        settings=json.loads((args.campaign_dir/model/'generation_settings.json').read_text(encoding='utf-8'))
        model_rows=[r for r in raw if r['model']==model]
        if len(model_rows)!=settings['count'] and not args.score_available:
            raise RuntimeError('Generation incomplete: '+model+'; use --annotation-only while generating')
        if settings['data_sha256']!=sha256(args.data): raise ValueError('Generation data hash differs from analysis data')
        if [r['index'] for r in model_rows]!=list(range(len(model_rows))): raise ValueError('Candidate indices are not complete/unique')
        for row in model_rows:
            expected_seed=settings['seed']+(row['index']//settings['batch_size'])*settings['batch_size']
            if row['batch_seed']!=expected_seed or row['rejection_reasons']!=rejection_reasons(row):
                raise ValueError('Candidate seed/basic-check provenance mismatch: '+row['id'])
        if model=='MAGE': common_settings=settings
        else:
            for key in ('count','batch_size','seed','top_p','temperature','max_new_tokens','antigen','data_sha256',
                        'checkpoint_sha256','progen_source_sha256','tokenizer_sha256'):
                if settings[key]!=common_settings[key]: raise ValueError('Models have different generation settings: '+key)
    args.seed=common_settings['seed']
    if not args.oasis_db.is_file(): raise FileNotFoundError('Official OASis database is not ready: '+str(args.oasis_db))
    verified=args.oasis_db.parent/'oasis_database_verified.json'
    if not verified.exists(): raise ValueError('OASis compressed download checksum verification record is missing')
    database_verification=json.loads(verified.read_text(encoding='utf-8'))
    if (database_verification.get('compressed_md5')!='28fa18135b3ef1710970f72b5953cbc5'
        or database_verification.get('compressed_bytes')!=6484781782
        or database_verification.get('db_bytes')!=args.oasis_db.stat().st_size):
        raise ValueError('Official OASis database verification record/file size mismatch')
    scoring_settings=dict(biophi_version=importlib.metadata.version('biophi'),min_fraction_subjects=.10,
                          database=database_verification,
                          database_path=str(args.oasis_db.resolve()),database_bytes=args.oasis_db.stat().st_size,
                          database_mtime_ns=args.oasis_db.stat().st_mtime_ns)
    scoring_manifest=output/'oasis_settings.json'
    if scoring_manifest.exists() and json.loads(scoring_manifest.read_text(encoding='utf-8'))!=scoring_settings:
        raise ValueError('OASis scoring/database differs from cached scores')
    write_json(scoring_manifest,scoring_settings)
    scores_path=output/'oasis_pair_cache.json'
    scores=json.loads(scores_path.read_text(encoding='utf-8')) if scores_path.exists() else {}
    valid=[r for r in annotated if r['annotation_pass']]
    tasks={}
    for row in valid:
        key=hashlib.sha256((row['vh']+'|'+row['vl']).encode()).hexdigest()
        if key not in scores:
            tasks[key]=(key,annotations[row['vh']],annotations[row['vl']],str(args.oasis_db.resolve()))
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    if args.oasis_workers<1: raise ValueError('oasis-workers must be positive')
    with ProcessPoolExecutor(max_workers=args.oasis_workers,mp_context=get_context('spawn')) as executor:
        for i,(key,score) in enumerate(executor.map(score_oasis_pair,tasks.values(),chunksize=1)):
            scores[key]=score
            if i%25==0:
                write_json(scores_path,scores);print(json.dumps(dict(phase='OASis',new_completed=i+1,new_total=len(tasks),cached_pairs=len(scores))),flush=True)
    write_json(scores_path,scores)
    for row in valid:
        key=hashlib.sha256((row['vh']+'|'+row['vl']).encode()).hexdigest()
        row.update(scores[key]);row['humanness_pass']=row['oasis_percentile']>=.70
    if args.score_available:
        write_json(output/'incremental_scoring_progress.json',dict(status='cached_only',attempts=len(raw),annotated=len(valid),new_scored=len(tasks),total_cached_pairs=len(scores)))
        print(json.dumps(dict(phase='partial_scoring_ready',attempts=len(raw),cached_pairs=len(scores))),flush=True);return
    with zipfile.ZipFile(ROOT/'MAGE_ab_generation/MAGE_annotated_training_data.zip') as archive:
        with archive.open('annotated_training_data.csv') as f: original=list(csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig')))
    references=dict(train85=train_valid,known106=known_valid,original=original)
    add_reference_metrics(valid,references)
    for refname,refrows in references.items():
        pairs={(r['VH_AA'],r['VL_AA']) for r in refrows if r.get('VH_AA') and r.get('VL_AA')}
        for row in valid: row['exact_pair_'+refname]=(row['VH_AA'],row['VL_AA']) in pairs
    all_rows=[];summary={};diversity={}
    for model in ('MAGE','LoRA'):
        attempts=[r for r in annotated if r['model']==model]
        numbered=[r for r in valid if r['model']==model]
        post=[r for r in numbered if r['humanness_pass']]
        selected,selection=select_paper_candidates(post)
        selection_by_id={s['id']:s['selection_route'] for s in selected}
        for row in attempts:
            row['humanness_pass']=bool(row.get('humanness_pass',False))
            row['selected']=row['id'] in selection_by_id
            row['selection_route']=selection_by_id.get(row['id'],'')
            triage=[]
            if not row['humanness_pass']: triage.append('preselection_not_passed')
            else:
                if row['v_identity.H']<.85: triage.append('VH_germline_identity_below_85')
                if row['v_identity.H']>=1: triage.append('VH_germline_identity_100')
                for ref in ('original','train85'):
                    if row['max_identity_'+ref+'_CDR3.H']>=1: triage.append('exact_HCDR3_'+ref)
            row['triage_reasons']=triage;row['triage_pass']=not triage
            row.pop('raw_ids',None);row.pop('raw_text',None)
        paircounts=Counter((r['VH_AA'],r['VL_AA']) for r in post)
        reasons=Counter(reason for r in attempts for reason in r['rejection_reasons'])
        summary[model]=dict(attempts=len(attempts),basic_pass=sum(not r['rejection_reasons'] for r in attempts),
             annotation_pass=len(numbered),oas70_pass=len(post),unique_post_pairs=len(paircounts),
             duplicate_post_attempts=len(post)-len(paircounts),basic_failure_reasons=dict(reasons),
             triage_pass_attempts=sum(r['triage_pass'] for r in attempts),
             triage_unique_pairs=selection['eligible_unique_pairs'],selected_pairs=selection['selected'],
             triage_reasons=dict(Counter(reason for r in attempts for reason in r['triage_reasons'])),
             annotation_failure_reasons=dict(Counter(reason for r in attempts for reason in r['annotation_reasons'] if reason not in r['rejection_reasons'])),
             selection=selection,
             median_oasis_before_threshold=statistics_median([r['oasis_percentile'] for r in numbered]),
             mean_HCDR3_identity_train85=statistics_mean([r['max_identity_train85_CDR3.H'] for r in post]),
             post_exact_pairs_train85=sum(r['exact_pair_train85'] for r in post),
             post_exact_HCDR3_train85=sum(r['max_identity_train85_CDR3.H']==1 for r in post),
             post_VH_identity_train85_ge90=sum(r['max_identity_train85_VH_AA']>=.9 for r in post))
        diversity[model]=within_generated_metrics(post)
        write_csv(output/(model+'_최종후보.csv'),selected)
        write_json(output/(model+'_최종후보.json'),selected)
        fasta=[]
        for r in selected:
            for c in ('H','L'): fasta.extend(['>'+r['id']+'_'+c+'|'+r['selection_route'],r['V'+c+'_AA']])
        (output/(model+'_최종후보.fasta')).write_text('\n'.join(fasta)+'\n',encoding='utf-8')
        write_selected_alignments(output,model,selected,known_valid)
        all_rows.extend(attempts)
    write_csv(output/'전체후보_단계별지표.csv',all_rows);write_json(output/'all_metrics.json',all_rows)
    write_json(output/'stage_summary.json',summary);write_json(output/'within_generated_diversity.json',diversity)
    packages={n:analysis_package_version(n) for n in ('abnumber','anarci','hmmer','biophi','numpy','pandas','SQLAlchemy','rapidfuzz','matplotlib')}
    write_json(output/'analysis_provenance.json',dict(seed=args.seed,python=sys.version,analysis_code_sha256=sha256(__file__),scheme='IMGT',allowed_species=['human'],
        min_fraction_subjects=.1,humanness_gate=.7,oasis_workers=args.oasis_workers,packages=packages,
        oasis_db=str(args.oasis_db),oasis_db_bytes=args.oasis_db.stat().st_size,
        original_data_sha256=sha256(ROOT/'MAGE_ab_generation/MAGE_annotated_training_data.zip'),
        reference_counts={k:len(v) for k,v in references.items()},known_annotation_failures=[r['id'] for r in known_annotated if not r['annotation_pass']],
        original_reference_available_counts={f:sum(bool(r.get(f)) for r in original) for f in ['VH_AA','VL_AA']+[f'{x}.{c}' for c in ('H','L') for x in REGIONS]},
        metrics='Full-chain/raw-region minimum Levenshtein and maximum 1-distance/max_length computed independently; figure F uses region distances to the SAME nearest full-chain reference (per VH/VL), lexicographic sequence tie break',
        selection_adaptations=['PD-L1 known106 replaces RBD SAbDab/CoV-AbDab','Exclude exact HCDR3 in original data AND LoRA train85','Deduplicate VH/VL pairs; no replacement for missing strata slots']))
    plot_generation_analysis(output,all_rows,summary,diversity,seed=args.seed)
    write_generation_report(output,summary)
    print(json.dumps(dict(status='analysis_complete',output=str(output),summary=summary),ensure_ascii=False),flush=True)


def statistics_mean(values):
    return sum(values)/len(values) if values else None


def statistics_median(values):
    import statistics
    return statistics.median(values) if values else None


def write_selected_alignments(output,model,selected,known):
    """최종 후보 HCDR3의 가장 가까운 알려진 양성과 편집거리 정렬을 표시한다."""
    from rapidfuzz.distance import Levenshtein
    from Bio.Align import PairwiseAligner
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    aligner=PairwiseAligner(mode='global',match_score=0,mismatch_score=-1,open_gap_score=-1,extend_gap_score=-1)
    lines=[];details=[]
    for row in selected:
        best=max(known,key=lambda r:Levenshtein.normalized_similarity(row['CDR3.H'],r['CDR3.H']))
        identity=Levenshtein.normalized_similarity(row['CDR3.H'],best['CDR3.H'])
        alignment=aligner.align(row['CDR3.H'],best['CDR3.H'])[0]
        lines.append(f"{row['id']} | {row['selection_route']} | known: {best['id']} | identity={identity:.3f}\n{alignment}")
        details.append(dict(candidate=row['id'],reference=best['id'],HCDR3=row['CDR3.H'],reference_HCDR3=best['CDR3.H'],identity=identity))
    (output/(model+'_선택후보_HCDR3정렬.txt')).write_text('\n'.join(lines),encoding='utf-8')
    write_csv(output/(model+'_선택후보_가까운양성.csv'),details)
    if lines:
        fig,ax=plt.subplots(figsize=(14,max(4,len(lines)*1.05)));ax.axis('off')
        ax.text(0,1,'\n'.join(lines),va='top',fontfamily='monospace',fontsize=8,transform=ax.transAxes)
        fig.suptitle(model+' selected candidates: HCDR3 vs nearest known PD-L1 positive')
        fig.savefig(output/(model+'_선택후보_HCDR3정렬.png'),dpi=160,bbox_inches='tight');plt.close(fig)


def write_generation_report(output,summary):
    lines=['# PD-L1 생성 항체 분석 결과','',
           '원본 MAGE와 학습된 LoRA를 같은 PD-L1 세포외 도메인으로 조건화했다. 시드와 생성 조건, 모델 해시는 모델별 generation_settings.json에 기록했다.',
           '','## 단계별 결과','']
    for model,s in summary.items():
        lines += [f"- {model}: 생성 {s['attempts']} → 기본검사 {s['basic_pass']} → ANARCI {s['annotation_pass']} → OASis ≥0.70 {s['oas70_pass']} → 고유 쌍 {s['unique_post_pairs']} → 추가 선별 {s['triage_pass_attempts']}회 / 최종 {s['selection']['selected']}쌍.",
                  f"  선별 전 OASis 중앙값: {s['median_oasis_before_threshold']}. 최종 20쌍 대비 부족: {s['selection']['shortage']}."]
    lines += ['', '## 그림을 읽는 방법','',
              '- 주요 A: 생성 실패와 각 단계 탈락을 포함한 후보 수. 기본 검사는 정상 아미노산·종료 토큰·한 번의 체인 경계·각 체인 길이를 확인한다.',
              '- 주요 B: OASis 기준을 적용하기 전, ANARCI 통과 집단의 paired percentile 분포. 선별 후 분포만 보여주는 편향을 피했다.',
              '- 주요 C/D/F: OASis 통과 집단과 LoRA 학습 85쌍 사이의 거리/유사도. 편집거리는 작을수록, identity는 클수록 학습 서열에 가깝다. F는 가장 가까운 전체 VH 또는 VL을 먼저 고른 뒤 같은 참조 서열의 각 영역과 비교한다. 오차막대는 집단 내 표준편차이며 반복 학습의 불확실성이 아니다.',
              '- 주요 E: 두 모델을 같은 VH/VL gene 축과 색상 범위로 비교. 값은 각 OASis 통과 집단에서의 비율이다.',
              '- 보충: 생성 집단 내부의 다양성, CDR3 길이, 원본 MAGE 학습 자료 대비 신규성 및 알려진 PD-L1 양성과의 근접성을 따로 표시했다.',
              '', '## 해석 범위','',
              '서열 분포에는 중복 생성도 포함하며 중복 수를 별도로 보고한다. 최종 후보는 쌍 단위 중복을 제거했다. ANARCI의 human germline 할당과 OASis 점수는 결합력이나 면역원성의 실험적 증거가 아니다. 알려진 양성 106쌍에는 검증셋도 포함하므로 known106 근접성은 독립적인 성능 평가가 아니다. 단일 seed에서의 생성 비교이며, 구조 예측·docking·결합 실험은 수행하지 않았다.',
              '', '논문 절차를 PD-L1에 적용했고 RBD 참조 양성을 PD-L1 106쌍으로 바꿨다. 정확한 기준과 작은 gene 그룹/중복의 처리 방식은 시작안내.md 및 analysis_provenance.json에 기록했다.', '']
    (output/'결과_읽는법.md').write_text('\n'.join(lines),encoding='utf-8')


def plot_generation_analysis(output,rows,summary,diversity,seed=42):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=('MAGE','LoRA');colors=('#65768c','#008b78')
    populations={m:[r for r in rows if r['model']==m and r.get('humanness_pass')] for m in models}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,2,figsize=(14,14),layout='constrained')
    ax=axes[0,0];stages=('attempts','basic_pass','annotation_pass','oas70_pass','unique_post_pairs','triage_unique_pairs','selected_pairs')
    x=np.arange(len(stages))
    for i,(m,color) in enumerate(zip(models,colors)):
        bars=ax.bar(x+(i-.5)*.36,[summary[m][s] for s in stages],.36,label=m,color=color)
        ax.bar_label(bars,padding=2,fontsize=8)
    ax.set_xticks(x,['Generated','Basic','IMGT','OASis >= .7','Unique pairs','Triage','Selected'],rotation=20,fontsize=8);ax.set_title('A  Candidate flow');ax.legend()
    ax=axes[0,1]
    for m,color in zip(models,colors):
        values=[r['oasis_percentile'] for r in rows if r['model']==m and r.get('annotation_pass')]
        if values: ax.hist(values,bins=np.linspace(0,1,21),histtype='step',linewidth=2,label=f'{m} (n={len(values)})',color=color)
    ax.axvline(.7,ls='--',color='black',label='Selection threshold');ax.set(xlabel='Official paired OASis percentile (10% subjects)',ylabel='Attempts',title='B  Humanness before thresholding');ax.legend(fontsize=8)
    ax=axes[1,0]
    for m,color in zip(models,colors):
        for chain,style in [('VH_AA','-'),('VL_AA','--')]:
            values=sorted(r['min_edit_train85_'+chain] for r in populations[m])
            if values: ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,ls=style,label=m+' '+chain[:2])
    ax.set(xlabel='Nearest LoRA-training sequence: edit distance',ylabel='Cumulative fraction',title='C  Novelty relative to train85');ax.legend(fontsize=8)
    ax=axes[1,1]
    for m,color in zip(models,colors):
        values=sorted(r['max_identity_train85_CDR3.H'] for r in populations[m])
        if values: ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=m,color=color)
    ax.set(xlabel='Maximum HCDR3 identity to train85',ylabel='Cumulative fraction',title='D  HCDR3 proximity to training');ax.set_xlim(0,1.02);ax.legend()
    # 하나의 공통 gene축/색상스케일로 두 모델을 비교한다.
    gene_grid=axes[2,0].get_subplotspec().subgridspec(1,2)
    axes[2,0].remove();gene_axes=[fig.add_subplot(gene_grid[0,i]) for i in range(2)]
    genesH=sorted({r['v_gene.H'] for p in populations.values() for r in p});genesL=sorted({r['v_gene.L'] for p in populations.values() for r in p})
    from collections import Counter
    matrices=[]
    for m in models:
        counts=Counter((r['v_gene.H'],r['v_gene.L']) for r in populations[m]);n=len(populations[m])
        matrices.append(np.array([[100*counts[(h,l)]/max(n,1) for l in genesL] for h in genesH]))
    if genesH and genesL:
        maximum=max(float(a.max()) for a in matrices)
        for ga,m,matrix in zip(gene_axes,models,matrices):
            im=ga.imshow(matrix,aspect='auto',vmin=0,vmax=maximum,cmap='viridis');ga.set_title('E  Germline usage: '+m,fontsize=9)
            ga.set_xticks(range(len(genesL)),genesL,rotation=90,fontsize=5);ga.set_yticks(range(len(genesH)),genesH,fontsize=5)
        fig.colorbar(im,ax=gene_axes,label='% of post-OASis attempts',shrink=.7)
    ax=axes[2,1];fields=[f'{r}.{c}' for c in ('H','L') for r in REGIONS]
    for i,(m,color) in enumerate(zip(models,colors)):
        values=[[r['chain_match_edit_train85_'+f] for r in populations[m]
                 if r['chain_match_edit_train85_'+f] is not None] for f in fields]
        means=[np.mean(v) if v else np.nan for v in values];sd=[np.std(v) if v else np.nan for v in values]
        ax.errorbar(np.arange(len(fields))+(i-.5)*.15,means,yerr=sd,marker='o',ms=3,capsize=2,color=color,label=m)
    ax.set_xticks(range(len(fields)),fields,rotation=90,fontsize=8);ax.set(ylabel='Nearest train85 edit distance (mean +/- SD)',title='F  Regionwise novelty');ax.legend()
    counts=' / '.join(f"{m}: {summary[m]['attempts']:,}" for m in models)
    postcounts=' / '.join(f'{m}: {len(populations[m])}' for m in models)
    fig.suptitle(f'PD-L1 antibody generation: original MAGE vs LoRA\nAttempts {counts} | seed {seed}\nPanels C-F: post-OASis attempts ({postcounts})',fontsize=15)
    fig.savefig(output/'생성항체_주요분석.png',dpi=220);fig.savefig(output/'생성항체_주요분석.svg');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    max_cdr3=max([len(r['CDR3.'+c]) for p in populations.values() for r in p for c in ('H','L')]+[1])
    for m,color in zip(models,colors):
        for chain,style in [('H','-'),('L','--')]:
            lengths=[len(r['CDR3.'+chain]) for r in populations[m]]
            if lengths: axes[0,0].hist(lengths,bins=np.arange(0,max_cdr3+2),histtype='step',ls=style,label=m+' '+chain,color=color)
    axes[0,0].set(title='A  CDR3 length',xlabel='Residues',ylabel='Attempts');axes[0,0].legend()
    for i,(m,color) in enumerate(zip(models,colors)):
        ds=diversity[m];axes[0,1].errorbar(np.arange(len(ds))+(i-.5)*.15,[r['mean'] if r['mean'] is not None else np.nan for r in ds],
            yerr=[r['std'] if r['std'] is not None else np.nan for r in ds],fmt='o',capsize=3,color=color,label=m)
    axes[0,1].set_xticks(range(8),[r['region'] for r in diversity['MAGE']],rotation=45);axes[0,1].set(title='B  Within-generated diversity',ylabel='Normalized pairwise edit distance (mean +/- SD)');axes[0,1].legend()
    for m,color in zip(models,colors):
        for chain,style in [('VH_AA','-'),('VL_AA','--')]:
            values=sorted(r['min_edit_original_'+chain] for r in populations[m])
            if values: axes[1,0].step(values,np.arange(1,len(values)+1)/len(values),where='post',ls=style,color=color,label=m+' '+chain[:2])
    axes[1,0].set(title='C  Novelty relative to original MAGE data',xlabel='Nearest original-training edit distance',ylabel='Cumulative fraction');axes[1,0].legend()
    for m,color in zip(models,colors):
        values=sorted(r['max_identity_known106_CDR3.H'] for r in populations[m])
        if values: axes[1,1].step(values,np.arange(1,len(values)+1)/len(values),where='post',label=m,color=color)
    axes[1,1].set(title='D  Proximity to known PD-L1 positives',xlabel='Maximum HCDR3 identity to known106',ylabel='Cumulative fraction');axes[1,1].legend()
    fig.suptitle('Supplementary sequence analysis | post-OASis attempts',fontsize=15)
    fig.savefig(output/'생성항체_보충분석.png',dpi=220);fig.savefig(output/'생성항체_보충분석.svg');plt.close(fig)


def preview_generation_results(args):
    """이미 저장된 점수만 즉시 표시한다. 모델 생성/주석/점수 계산은 하지 않는다."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    folder=args.campaign_dir/'분석'
    rows=json.loads((folder/'generated_annotated.json').read_text(encoding='utf-8'))
    scores=json.loads((folder/'oasis_pair_cache.json').read_text(encoding='utf-8'))
    models=('MAGE','LoRA');colors=('#65768c','#008b78');summary={};scored={}
    for model in models:
        raw=campaign_records(args.campaign_dir/model)
        ready=[]
        for row in rows:
            if row['model']!=model or not row['annotation_pass']: continue
            key=hashlib.sha256((row['vh']+'|'+row['vl']).encode()).hexdigest()
            if key in scores: ready.append(dict(row,**scores[key]))
        scored[model]=ready
        snapshot=sum(r['model']==model for r in rows)
        summary[model]=dict(generated_attempts=len(raw),snapshot_attempts=snapshot,
             scored_attempts=len(ready),oas70_pass=sum(r['oasis_percentile']>=.7 for r in ready),
             median_oasis=statistics_median([r['oasis_percentile'] for r in ready]),
             not_yet_in_analysis_snapshot=len(raw)-snapshot)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for model,color in zip(models,colors):
        data=scored[model]
        axes[0].hist([r['oasis_percentile'] for r in data],bins=np.linspace(0,1,21),
                     density=True,histtype='step',lw=2,color=color,label=f'{model} (scored n={len(data)})')
        passed=[r for r in data if r['oasis_percentile']>=.7]
        lengths=[len(r['CDR3.H']) for r in passed]
        if lengths: axes[1].hist(lengths,bins=np.arange(0,52),density=True,histtype='step',lw=2,
                                color=color,label=f'{model} (post-OASis n={len(passed)})')
    axes[0].axvline(.7,color='black',ls='--',lw=1,label='OASis threshold .70')
    axes[0].set(title='A  Humanness before selection',xlabel='Official paired OASis percentile',ylabel='Density')
    axes[1].set(title='B  HCDR3 length after OASis selection',xlabel='HCDR3 residues',ylabel='Density')
    for ax in axes:
        ax.legend(fontsize=8);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    counts=' / '.join(f"{m} generated {summary[m]['generated_attempts']}/1000" for m in models)
    fig.suptitle('INTERIM RESULTS: cached analysis only\n'+counts+' | scored sample sizes shown in legends',fontsize=12)
    fig.savefig(folder/'중간미리보기.png',dpi=180);fig.savefig(folder/'중간미리보기.svg');plt.close(fig)
    write_json(folder/'중간결과_요약.json',summary)
    examples=sorted(scored['LoRA'],key=lambda r:(-r['oasis_percentile'],r['id']))[:10]
    write_csv(folder/'LoRA_중간예시10쌍.csv',[{k:r[k] for k in ('id','VH_AA','VL_AA','oasis_percentile','CDR3.H','v_gene.H','v_gene.L')} for r in examples])
    (folder/'LoRA_중간예시10쌍.fasta').write_text('\n'.join(f">{r['id']}_{c}|cached_OASis={r['oasis_percentile']:.4f}\n{r['V'+c+'_AA']}" for r in examples for c in ('H','L'))+'\n',encoding='utf-8')
    print(json.dumps(dict(status='interim_preview_only',summary=summary,figure=str(folder/'중간미리보기.png')),ensure_ascii=False),flush=True)


def linux_analysis(args):
    def linux_path(path):
        p=Path(path).resolve();return '/mnt/'+p.drive[0].lower()+p.as_posix()[2:]
    command=['wsl','-d','Ubuntu-24.04','--exec',ANALYSIS_PYTHON,linux_path(__file__),
             '--mode','annotate-worker','--campaign-dir',linux_path(args.campaign_dir),
             '--data',linux_path(args.data),'--split',linux_path(args.split),'--oasis-db',linux_path(args.oasis_db),
             '--seed',str(args.seed)]
    if args.annotation_only: command.append('--annotation-only')
    if args.score_available: command.append('--score-available')
    command.extend(['--oasis-workers',str(args.oasis_workers)])
    subprocess.run(command,check=True)


# ── 7. 실행 메뉴: 평소에는 이 파일과 맨 위 설정만 확인 ──────────────
def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['generate','evaluate','train','split','plot','summary','check','generate-many','campaign','analyze','annotate-worker','preview'],default=CONFIG['mode'])
    parser.add_argument('--count',type=int,default=1000)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--annotation-only',action='store_true',help='OASis 점수 없이 ANARCI 주석만 캐시')
    parser.add_argument('--score-available',action='store_true',help='완료된 배치만 주석/OASis 캐시; 최종 선별·그림은 만들지 않음')
    parser.add_argument('--oasis-workers',type=int,default=4)
    parser.add_argument('--campaign-dir',type=Path,default=HERE/'생성결과/PDL1_생성비교_20261111_시드42')
    parser.add_argument('--oasis-db',type=Path,default=HERE/'캐시/분석도구/OASis_9mers_v1.db')
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--split',type=Path,default=DEFAULT_SPLIT)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--adapter',type=Path,default=DEFAULT_ADAPTER)
    parser.add_argument('--antigen',help='미지정 시 CSV의 유일한 항원 서열')
    parser.add_argument('--seed','--train-seed',dest='seed',type=int,default=CONFIG['seed'])
    parser.add_argument('--split-seed',type=int,default=SPLIT_CONFIG['seed'])
    parser.add_argument('--threshold',type=float,default=SPLIT_CONFIG['threshold'])
    parser.add_argument('--val-fraction',type=float,default=SPLIT_CONFIG['val_fraction'])
    parser.add_argument('--epochs',type=int,default=TRAIN_CONFIG['max_epochs'])
    parser.add_argument('--top-p',type=float,default=CONFIG['top_p'])
    parser.add_argument('--temperature',type=float,default=CONFIG['temperature'])
    parser.add_argument('--max-new-tokens',type=int,default=CONFIG['max_new_tokens'])
    parser.add_argument('--output',type=Path,help='새 학습/분할/생성/평가 폴더; 미지정 시 날짜별 생성')
    parser.add_argument('--output-dir',type=Path,help='시각화 저장 폴더')
    parser.add_argument('--show',action='store_true',help='시각화 창도 표시')
    return parser


def main():
    parser=build_parser();args=parser.parse_args()
    if args.mode in ('generate-many','campaign') and not any(s.split('=')[0]=='--max-new-tokens' for s in sys.argv):
        args.max_new_tokens=250
    if args.epochs<1: parser.error('--epochs must be positive')
    if args.annotation_only and args.score_available: parser.error('Use either annotation-only or score-available')
    if args.mode=='annotate-worker':
        analysis_worker(args);return
    use_verified_python(args.seed)
    if args.mode=='preview':
        preview_generation_results(args)
    elif args.mode=='analyze':
        linux_analysis(args)
    elif args.mode=='campaign':
        # 모델별 프로세스 수명으로 VRAM을 반환한다. 완료 배치는 자동 재사용한다.
        for script,label in [(HERE/'MAGE.py','MAGE'),(Path(__file__),'LoRA')]:
            command=[str(PYTHON),'-X','utf8','-u',str(script),'--mode','generate-many',
                     '--count',str(args.count),'--batch-size',str(args.batch_size),'--seed',str(args.seed),
                     '--top-p',str(args.top_p),'--temperature',str(args.temperature),
                     '--max-new-tokens',str(args.max_new_tokens),'--output',str(args.campaign_dir/label),
                     '--data',str(args.data)]
            if label=='LoRA': command.extend(['--adapter',str(args.adapter)])
            if args.antigen: command.extend(['--antigen',args.antigen])
            subprocess.run(command,check=True)
        linux_analysis(args)
    elif args.mode=='generate-many':
        generate_many(args,args.adapter)
    elif args.mode=='plot':
        plot_results(args)
    elif args.mode=='summary':
        result=json.loads((args.run_dir/'result.json').read_text(encoding='utf-8'))
        print(json.dumps({k:result[k] for k in ('status','epochs_completed','best_epoch',
                         'baseline_validation','best_validation','peak_allocated_gib')},ensure_ascii=False,indent=2))
        print('사용할 어댑터: '+str(args.adapter))
    elif args.mode=='check':
        packages=check_environment(('peft','rapidfuzz','matplotlib'))
        manifest,datasets=load_verified_split(args.data,args.split)
        if not (args.adapter/'adapter_model.safetensors').is_file():
            raise FileNotFoundError('최적 어댑터 경로를 확인하세요: '+str(args.adapter))
        print(json.dumps(dict(packages=packages,counts={k:len(v) for k,v in datasets.items()},
                              checkpoint=str(CHECKPOINT),adapter=str(args.adapter)),ensure_ascii=False,indent=2))
    elif args.mode=='split':
        check_environment(('rapidfuzz',))
        output=args.output or HERE/'학습결과'/new_output('데이터분할').name
        result=prepare_split(args.data,output,args.split_seed,args.threshold,args.val_fraction)
        print(json.dumps(result['counts'],indent=2));print('저장: '+str(output))
    elif args.mode=='train':
        check_environment(('peft','rapidfuzz','matplotlib'))
        args.train_seed=args.seed
        if not args.split.exists():
            prepare_split(args.data,args.split,args.split_seed,args.threshold,args.val_fraction)
        args.output=args.output or HERE/'학습결과'/new_output('본학습').name
        args.output.mkdir(parents=True,exist_ok=False)
        try:
            train_model(args)
            args.run_dir=args.output
            plot_results(args)
        except BaseException as error:
            # 시각화 실패가 이미 끝난 학습 결과를 failed로 덮어쓰지 않도록 구분한다.
            if not (args.output/'result.json').exists():
                write_json(args.output/'result.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc()))
            else:
                write_json(args.output/'postprocess_error.json',dict(error=repr(error),traceback=traceback.format_exc()))
            raise
    elif args.mode=='evaluate':
        load_verified_split(args.data,args.split)
        evaluate(args,args.adapter)
    else:
        generate(args,args.adapter)


if __name__=='__main__': main()

