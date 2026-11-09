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
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

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


# ── 6. 실행 메뉴: 평소에는 이 파일과 맨 위 설정만 확인 ──────────────
def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['generate','evaluate','train','split','plot','summary','check'],default=CONFIG['mode'])
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
    if args.epochs<1: parser.error('--epochs must be positive')
    use_verified_python(args.seed)
    if args.mode=='plot':
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

