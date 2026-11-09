"""원본 MAGE: 이 파일 하나에서 설정 -> 모델 로딩 -> 생성/NLL 평가.

그냥 실행: PD-L1 항체 1쌍 생성
python MAGE.py --mode evaluate : 고정 검증셋 21쌍 NLL 평가
python MAGE.py --mode check    : 경로/의존성 확인 (모델 가중치 로딩 없음)
"""
import argparse
import csv
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
CONFIG = dict(mode='generate', seed=42, top_p=0.9, temperature=1.0, max_new_tokens=256)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HOME'] = str(HERE/'캐시/huggingface')
os.environ['TRANSFORMERS_CACHE'] = str(HERE/'캐시/huggingface/hub')
os.environ['HF_HUB_OFFLINE'] = '1'
AA = set('ACDEFGHIKLMNPQRSTVWY')


# ── 2. 실행환경·파일 입출력 ──────────────────────────────────────
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


# ── 3. 모델 로딩: 외부 ProGen은 설치 라이브러리처럼 사용 ─────────────
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


# ── 4. 생성: 항원 -> [SEP] -> VH -> [LC] -> VL ─────────────────────
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


# ── 5. 동일 검증 서열의 조건부 NLL (정답 토큰을 주는 teacher forcing) ──
def encode_example(row,tokenizer,max_length=None):
    ids=tokenizer.encode(row['SEQ'],add_special_tokens=False)
    sep=tokenizer.encode('[SEP]',add_special_tokens=False)
    lc=tokenizer.encode('[LC]',add_special_tokens=False)
    if len(sep)!=1 or len(lc)!=1 or ids.count(sep[0])!=1 or ids.count(lc[0])!=1:
        raise ValueError('구분 토큰이 잘못됐습니다.')
    ids=ids+[tokenizer.eos_token_id]
    labels=ids.copy();boundary=ids.index(sep[0])+1;labels[:boundary]=[-100]*boundary
    mask=[1]*len(ids)
    if max_length is not None:
        if len(ids)>max_length: raise ValueError('Refusing to truncate antibody example')
        padding=max_length-len(ids)
        ids += [tokenizer.pad_token_id]*padding;labels += [-100]*padding;mask += [0]*padding
    return dict(input_ids=ids,attention_mask=mask,labels=labels)


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


# ── 6. 실행: 위 함수들을 이곳에서 연결 ─────────────────────────────
def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['generate','evaluate','check'],default=CONFIG['mode'])
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--split',type=Path,default=DEFAULT_SPLIT)
    parser.add_argument('--antigen',help='미지정 시 CSV의 유일한 항원 서열 사용')
    parser.add_argument('--seed',type=int,default=CONFIG['seed'])
    parser.add_argument('--top-p',type=float,default=CONFIG['top_p'])
    parser.add_argument('--temperature',type=float,default=CONFIG['temperature'])
    parser.add_argument('--max-new-tokens',type=int,default=CONFIG['max_new_tokens'])
    parser.add_argument('--output',type=Path,help='새 결과 폴더; 미지정 시 날짜별 자동 생성')
    return parser


def main():
    args=build_parser().parse_args();use_verified_python(args.seed)
    if args.mode=='check':
        print(json.dumps(dict(packages=check_environment(),checkpoint=str(CHECKPOINT),source=str(SOURCE)),ensure_ascii=False,indent=2))
    elif args.mode=='evaluate': evaluate(args)
    else: generate(args)


if __name__=='__main__': main()
