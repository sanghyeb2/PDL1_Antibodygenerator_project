"""PD-L1 생성 항체 구조 평가. 준비/예측/도킹/분석을 이 파일에서 이어 실행합니다.

Linux/WSL 실행 권장. prepare는 기존 ANARCI 환경, fold는 ColabFold 환경 사용.
예: python 구조평가.py prepare
    python 구조평가.py fold --limit 1
전체 설정 및 입력 해시는 구조결과/.../manifest.json에 보존됩니다.
"""
from __future__ import annotations

import argparse
import hashlib
import csv
import gzip
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
SOURCE = HERE / '공유결과' / 'PDL1_MAGE_LoRA_20261114'
OUTPUT = Path(os.environ.get('PDL1_STRUCTURE_OUTPUT', HERE / '구조결과' / 'PDL1_40쌍_대조군_20261114_시드42'))
ENV = Path(os.environ.get('PDL1_STRUCTURE_ENV', '/root/.cache/pdl1-structure'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url, path):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.download')
    urllib.request.urlretrieve(url, temp)
    temp.replace(path)


def prepare():
    from abnumber import Chain
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    candidates = []
    hashes = {}
    for group in ('MAGE', 'LoRA'):
        path = SOURCE / '분석' / f'{group}_최종후보.json'
        rows = read_json(path)
        assert len(rows) == 20, (group, len(rows))
        hashes[str(path.relative_to(HERE))] = digest(path)
        for row in rows:
            candidates.append({key: row[key] for key in ('id', 'VH_AA', 'VL_AA')}
                              | {'group': group, 'regions': {k: v for k, v in row.items()
                                 if k.startswith(('CDR', 'FWR'))}})
    for pdb_id, label in [('5X8L', 'atezolizumab'), ('5X8M', 'durvalumab')]:
        cif = OUTPUT / 'references' / f'{pdb_id}.cif'
        download(f'https://files.rcsb.org/download/{pdb_id}.cif', cif)
        download(f'https://files.rcsb.org/download/{pdb_id}.pdb', cif.with_suffix('.pdb'))
        data = MMCIF2Dict(str(cif))
        domains = {}
        for seq, chains in zip(data['_entity_poly.pdbx_seq_one_letter_code_can'],
                               data['_entity_poly.pdbx_strand_id']):
            seq = ''.join(seq.split())
            try:
                chain = Chain(seq, scheme='imgt')
            except Exception:
                continue  # PD-L1 and non-immunoglobulin entities are not Fv chains.
            kind = 'H' if chain.is_heavy_chain() else 'L'
            if kind in domains:
                raise ValueError(f'Ambiguous variable-domain entity in {pdb_id}')
            domains[kind] = {'sequence': chain.seq, 'source_chains': chains.split(','),
                             'entity_sequence': seq, 'domain_start': seq.index(chain.seq),
                             'regions': {f'CDR{i}.{kind}': getattr(chain, f'cdr{i}_seq') for i in (1, 2, 3)}}
        assert set(domains) == {'H', 'L'}, (pdb_id, domains.keys())
        candidates.append({'id': f'CTRL_{label}', 'group': 'control',
                           'VH_AA': domains['H']['sequence'], 'VL_AA': domains['L']['sequence'],
                           'regions': domains['H']['regions'] | domains['L']['regions'],
                           'reference_pdb': pdb_id, 'reference_domains': domains})
        hashes[f'references/{pdb_id}.cif'] = digest(cif)
    antigen_settings = SOURCE / 'LoRA' / 'generation_settings.json'
    antigen = read_json(antigen_settings)['antigen']
    assert len(antigen) == 220
    hashes[str(antigen_settings.relative_to(HERE))] = digest(antigen_settings)
    settings = {'seed': 42, 'fold': {'model_type': 'alphafold2_multimer_v3', 'num_models': 5,
                 'num_recycle': 3, 'num_seeds': 1, 'msa_mode': 'mmseqs2_uniref_env',
                 'pair_mode': 'unpaired', 'templates': False, 'amber': True,
                 'max_msa': '256:512', 'XLA_FLAGS': '--xla_gpu_autotune_level=0'},
                'docking': {'epitope': 'whole solvent-accessible PD-L1 ECD',
                            'paratope': 'six IMGT CDRs', 'final_budget': None,
                            'status': 'awaiting measured pilot'},
                'prodigy': {'temperature_C': 25, 'distance_cutoff_A': 5.5,
                            'selection': ['H,L', 'A']},
                'interface_contact_cutoff_A': 4.0,
                'uniprot': 'Q9NZQ7', 'antigen_uniprot_range': [19, 238],
                'glycosylation_sites_uniprot': [35, 192, 200, 219]}
    manifest = {'schema': 1, 'input_hashes': hashes, 'settings': settings,
                'antigen': antigen, 'candidates': candidates}
    path = OUTPUT / 'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise RuntimeError('Existing manifest differs; choose a new output directory.')
    write_json(path, manifest)
    inputs = OUTPUT / 'inputs'
    inputs.mkdir(exist_ok=True)
    for row in candidates:
        (inputs / f"{row['id']}.fasta").write_text(
            f">{row['id']}\n{row['VH_AA']}:{row['VL_AA']}\n", encoding='ascii')
    (inputs / 'PDL1_ECD.fasta').write_text(f'>PDL1_ECD\n{antigen}\n', encoding='ascii')
    write_json(OUTPUT / 'preparation_summary.json', {
        'candidate_count': len(candidates), 'MAGE': 20, 'LoRA': 20, 'positive_controls': 2,
        'Fv_chain_lengths': {r['id']: [len(r['VH_AA']), len(r['VL_AA'])] for r in candidates},
        'antigen_length': len(antigen), 'seed': 42})
    print(f'Prepared {len(candidates)} Fv + common antigen: {OUTPUT}', flush=True)


def run_logged(command, log, cwd=None, extra_env=None, monitor_gpu=False, on_poll=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metadata = {'command': [str(x) for x in command], 'cwd': str(cwd or Path.cwd()),
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'environment_overrides': extra_env or {}, 'runner_sha256': digest(__file__)}
    write_json(log.with_suffix('.command.json'), metadata)
    with log.open('w', encoding='utf-8') as handle:
        process = subprocess.Popen([str(x) for x in command], cwd=cwd, stdout=handle,
                                   stderr=subprocess.STDOUT, env=os.environ | (extra_env or {}))
        metadata['pid'] = process.pid
        write_json(log.with_suffix('.command.json'), metadata)
        gpu_samples = []
        while True:
            if on_poll:
                on_poll()
            if monitor_gpu:
                sample = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True)
                if sample.returncode == 0:
                    gpu_samples.append(sample.stdout.strip())
            try:
                result = process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                pass
        if gpu_samples:
            log.with_suffix('.gpu.csv').write_text(
                'timestamp,memory_used_MiB,gpu_utilization_percent\n' + '\n'.join(gpu_samples) + '\n')
            metadata['peak_observed_gpu_memory_MiB'] = max(
                int(x.split(',')[1]) for x in gpu_samples)
    metadata.update(exit_code=result, elapsed_seconds=time.time() - started)
    write_json(log.with_suffix('.command.json'), metadata)
    if result:
        raise RuntimeError(f'Command failed ({result}); see {log}')


def fold(limit=None, ids=None):
    manifest = read_json(OUTPUT / 'manifest.json')
    rows = manifest['candidates']
    # Antigen first, then balanced pilot candidates, then the rest.
    initial = ['PDL1_ECD', rows[0]['id'], rows[20]['id'], 'CTRL_atezolizumab', 'CTRL_durvalumab']
    order = initial + [r['id'] for r in rows if r['id'] not in initial]
    if ids:
        order = ids
    if limit:
        order = order[:limit]
    executable = ENV / 'colabfold/bin/colabfold_batch'
    for candidate in order:
        fasta = OUTPUT / 'inputs' / f'{candidate}.fasta'
        destination = OUTPUT / 'folding' / candidate
        destination.mkdir(parents=True, exist_ok=True)
        signature = {'input_sha256': digest(fasta), 'settings': manifest['settings']['fold'],
                     'seed': manifest['settings']['seed'], 'antigen_monomer': candidate == 'PDL1_ECD'}
        done = destination / 'completed.json'
        if done.exists():
            old = read_json(done)
            if old['signature'] != signature:
                raise RuntimeError(f'Cached folding settings differ: {candidate}')
            if not all((destination / p).exists() for p in old['rank1_pdbs']):
                raise RuntimeError(f'Incomplete cached PDB files: {candidate}')
            print(f'Cached: {candidate}', flush=True)
            continue
        command = [executable, fasta, destination, '--model-type',
                   'alphafold2_ptm' if candidate == 'PDL1_ECD' else 'alphafold2_multimer_v3',
                   '--num-models', '5', '--num-recycle', '3', '--num-seeds', '1',
                   '--random-seed', '42', '--pair-mode', 'unpaired', '--max-msa', '256:512',
                   '--amber', '--num-relax', '1']
        print(f'Folding: {candidate}', flush=True)
        # Preserve failed attempts before resuming cached MSA/weights.
        for old_name in ('execution.log', 'execution.command.json', 'execution.gpu.csv'):
            old = destination / old_name
            if old.exists():
                attempts = destination / 'attempts'
                attempts.mkdir(exist_ok=True)
                shutil.copy2(old, attempts / f'{time.time_ns()}_{old_name}')
        run_logged(command, destination / 'execution.log', extra_env={
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
            'XLA_FLAGS': '--xla_gpu_autotune_level=0',
            'TF_ENABLE_ONEDNN_OPTS': '0',
            'PYTHONFAULTHANDLER': '1',
            'MPLBACKEND': 'Agg',
            'PATH': str(executable.parent) + ':' + os.environ.get('PATH', '')}, monitor_gpu=True)
        ranked = list(destination.glob('*_relaxed_rank_001*.pdb'))
        if not ranked or not list(destination.glob('*_scores_rank_001*.json')):
            raise RuntimeError(f'No rank-1 structure and confidence output: {candidate}')
        write_json(done, {'signature': signature, 'rank1_pdbs': [p.name for p in ranked],
                          'runtime_environment': {'XLA_FLAGS': '--xla_gpu_autotune_level=0',
                                                   'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'}})


def fold_batch(ids=None):
    """Keep one AF process for all Fv sequences to reuse padded-shape JIT compilation."""
    manifest = read_json(OUTPUT / 'manifest.json')
    rows = manifest['candidates']
    if ids:
        rows = [r for r in rows if r['id'] in ids]
        if len(rows) != len(ids):
            raise ValueError('Unknown or repeated candidate IDs')
    pending = []
    for row in rows:
        done = OUTPUT / 'folding' / row['id'] / 'completed.json'
        signature = {'input_sha256': digest(OUTPUT / 'inputs' / f"{row['id']}.fasta"),
                     'settings': manifest['settings']['fold'], 'seed': 42, 'antigen_monomer': False}
        if done.exists():
            if read_json(done)['signature'] != signature:
                raise RuntimeError(f'Folding cache mismatch: {row["id"]}')
            continue
        pending.append((row, signature))
    if not pending:
        print('All requested Fv structures already complete', flush=True)
        return
    batch = OUTPUT / 'folding' / 'Fv_batch'
    batch.mkdir(parents=True, exist_ok=True)
    settings = {'fold': manifest['settings']['fold'], 'seed': 42,
                'model_type': 'alphafold2_multimer_v3', 'recompile_padding': 10}
    if (batch / 'settings.json').exists() and read_json(batch / 'settings.json') != settings:
        raise RuntimeError('Batch settings differ from cached run')
    write_json(batch / 'settings.json', settings)
    csv_path = OUTPUT / 'inputs' / 'Fv_batch.csv'
    with csv_path.open('w', encoding='ascii', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'sequence'])
        for row, _ in pending:
            writer.writerow([row['id'], row['VH_AA'] + ':' + row['VL_AA']])

    def collect():
        for row, signature in pending:
            candidate = row['id']
            directory = OUTPUT / 'folding' / candidate
            if (directory / 'completed.json').exists() or not (batch / f'{candidate}.done.txt').exists():
                continue
            pdbs = list(batch.glob(f'{candidate}_relaxed_rank_001*.pdb'))
            scores = list(batch.glob(f'{candidate}_scores_rank_001*.json'))
            if not pdbs or not scores:
                continue
            directory.mkdir(exist_ok=True)
            paths = list(batch.glob(candidate + '_*')) + list(batch.glob(candidate + '.*'))
            for path in paths:
                if path.is_file():
                    shutil.copy2(path, directory / path.name)
            write_json(directory / 'completed.json', {'signature': signature,
                       'rank1_pdbs': [p.name for p in pdbs], 'batch': str(batch)})
            print(f'Completed Fv: {candidate}', flush=True)
            status()

    executable = ENV / 'colabfold/bin/colabfold_batch'
    command = [executable, csv_path, batch, '--model-type', 'alphafold2_multimer_v3',
               '--num-models', '5', '--num-recycle', '3', '--num-seeds', '1', '--random-seed', '42',
               '--pair-mode', 'unpaired', '--max-msa', '256:512', '--recompile-padding', '10',
               '--amber', '--num-relax', '1']
    run_logged(command, batch / 'execution.log', extra_env={
        'PATH': str(executable.parent) + ':' + os.environ.get('PATH', ''),
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'XLA_FLAGS': '--xla_gpu_autotune_level=0',
        'TF_ENABLE_ONEDNN_OPTS': '0', 'PYTHONFAULTHANDLER': '1', 'MPLBACKEND': 'Agg'}, monitor_gpu=True, on_poll=collect)
    collect()
    missing = [r['id'] for r, _ in pending if not (OUTPUT / 'folding' / r['id'] / 'completed.json').exists()]
    if missing:
        raise RuntimeError(f'Fv batch incomplete; inspect ColabFold log: {missing}')


def cdr_positions(row):
    """Map annotated CDR sequences to one-based sequence indices, ordered along each chain."""
    positions = {}
    for kind, key in [('H', 'VH_AA'), ('L', 'VL_AA')]:
        seq = row[key]
        offset = 0
        for number in (1, 2, 3):
            name = f'CDR{number}.{kind}'
            cdr = row['regions'][name]
            start = seq.find(cdr, offset)
            if start < 0 or not cdr:
                raise ValueError(f"Cannot map {row['id']} {name}")
            positions[name] = list(range(start + 1, start + len(cdr) + 1))
            offset = start + len(cdr)
    return positions


def predicted_structure(candidate):
    directory = OUTPUT / 'folding' / candidate
    done = read_json(directory / 'completed.json')
    return directory / done['rank1_pdbs'][0]


def write_partner(source, sequences, destination):
    """Rename AF chains, globally number the two Fv chains, and record all mappings."""
    from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain
    from Bio.SeqUtils import seq1
    model = PDBParser(QUIET=True).get_structure('fold', source)[0]
    source_chains = list(model.get_chains())
    if len(source_chains) != len(sequences):
        raise ValueError(f'Unexpected chains in {source}')
    output = Structure.Structure('partner')
    output.add(Model.Model(0))
    mapping = {}
    counter = 0
    for chain, (label, sequence) in zip(source_chains, sequences.items()):
        residues = [r for r in chain if r.id[0] == ' ']
        observed = ''.join(seq1(r.resname) for r in residues)
        if observed != sequence:
            raise ValueError(f'Folded sequence mismatch: {source} chain {chain.id}')
        target = Chain.Chain(label)
        output[0].add(target)
        mapping[label] = {}
        for index, residue in enumerate(residues, 1):
            counter += 1
            new = residue.copy()
            new.id = (' ', counter, ' ')
            new.segid = label
            target.add(new)
            mapping[label][str(counter)] = {'sequence_index': index, 'aa': sequence[index-1],
                                          'uniprot': index + 18 if label == 'A' else None}
    writer = PDBIO()
    writer.set_structure(output)
    writer.save(str(destination))
    return mapping


def prepare_docking(row, sampling, refine, profile, native_reference=False, reference_source=None, controlled=None):
    from haddock.clis.restraints.calc_accessibility import get_accessibility, apply_cutoff
    root = ENV / 'runs' / profile / row['id']
    root.mkdir(parents=True, exist_ok=True)
    antibody = root / 'antibody.pdb'
    antigen = root / 'antigen.pdb'
    if native_reference:
        if row['group'] != 'control' or not profile.startswith('reference-pilot'):
            raise ValueError('Native reference inputs are reserved for the separate reference-pilot profile')
        from Bio.PDB import PDBParser, PDBIO, Select
        source = Path(reference_source) if reference_source else OUTPUT / 'references' / row['id'] / 'resolved_Fv_PDL1.pdb'
        model = PDBParser(QUIET=True).get_structure('native', source)
        class Chains(Select):
            def __init__(self, names):
                self.names = names
            def accept_chain(self, chain):
                return chain.id in self.names
        writer = PDBIO()
        writer.set_structure(model)
        writer.save(str(antibody), Chains({'H', 'L'}))
        writer.save(str(antigen), Chains({'A'}))
        maps = {'input_scope': 'Resolved experimental Fv/IgV only; separate redocking pilot',
                'source': str(source)}
        if reference_source:
            maps['input_scope'] = 'ColabFold coordinates cropped to experimental resolved residue IDs; validation only'
    else:
        maps = {'antibody': write_partner(predicted_structure(row['id']),
                                         {'H': row['VH_AA'], 'L': row['VL_AA']}, antibody),
                'antigen': write_partner(predicted_structure('PDL1_ECD'),
                                        {'A': read_json(OUTPUT / 'manifest.json')['antigen']}, antigen)}
    cdrs = cdr_positions(row)
    accessible_fv = apply_cutoff(get_accessibility(str(antibody)), 0.4)
    accessible_ag = apply_cutoff(get_accessibility(str(antigen)), 0.4)['A']
    cdrs_global = {name: [i + (len(row['VH_AA']) if name.endswith('.L') else 0)
                        for i in values] for name, values in cdrs.items()}
    active = {chain: sorted(set(i for name, values in cdrs_global.items()
                               if name.endswith('.' + chain) for i in values)
                           & set(accessible_fv[chain])) for chain in ('H', 'L')}
    if controlled:
        if not native_reference or not profile.startswith('reference-pilot-controlled-'):
            raise ValueError('Controlled policy is reserved for separate control-validation profiles')
        fixed = controlled['fixed_AIR']
        for chain in ('H', 'L'):
            if not set(fixed[chain]) <= set(active[chain]):
                raise ValueError(f'Fixed CDR restraints are not exposed in this input: {chain}')
        if not set(fixed['A']) <= set(accessible_ag):
            raise ValueError('Fixed antigen restraints are not exposed in this input')
        active = {chain: sorted(fixed[chain]) for chain in ('H', 'L')}
        accessible_ag = sorted(fixed['A'])
    if not accessible_ag or not all(active.values()):
        raise ValueError(f"No exposed antigen/CDR residues for {row['id']}")
    # Official HADDOCK representation: one segid for the Fv docking body, with
    # a real backbone break between VH and VL. No linker residues are added.
    # topoaa/prot_break.cns removes the spurious inter-domain peptide link;
    # restrain_bodies keeps the two disconnected domains assembled in flexref.
    shutil.copy2(antibody, root / 'antibody_HL.pdb')
    merged = []
    for line in antibody.read_text().splitlines():
        if line.startswith('ATOM'):
            merged.append(line[:21] + 'B' + line[22:72].ljust(50) + 'B'.ljust(4) + line[76:])
    antibody.write_text('\n'.join(merged) + '\nEND\n', encoding='ascii')
    maps['docking_Fv_chain'] = 'B'
    maps['VH_end_global_residue'] = len(row['VH_AA'])
    heavy_ids = [int(line[22:26]) for line in (root / 'antibody_HL.pdb').read_text().splitlines()
                 if line.startswith('ATOM') and line[21] == 'H']
    light_ids = [int(line[22:26]) for line in (root / 'antibody_HL.pdb').read_text().splitlines()
                 if line.startswith('ATOM') and line[21] == 'L']
    maps['backbone_break'] = [max(heavy_ids), min(light_ids)]
    # CNS truncates long input lines: put each residue alternative on its own line.
    ag_selection = '\n         or '.join(f'resid {i}' for i in accessible_ag)
    lines = [f'assign (segid B and resid {i})\n'
             f'       (segid A and ({ag_selection})) 2.0 2.0 0.0\n'
             for chain in ('H', 'L') for i in active[chain]]
    (root / 'ambig.tbl').write_text('\n'.join(lines), encoding='ascii')
    restraints = subprocess.run([str(ENV / 'docking/bin/haddock3-restraints'),
                                  'restrain_bodies', str(antibody)], capture_output=True, text=True, check=True)
    if 'assign' not in restraints.stdout.lower():
        raise RuntimeError('No Fv body restraints generated')
    (root / 'unambig.tbl').write_text(restraints.stdout, encoding='ascii')
    write_json(root / 'residue_map.json', maps | {'cdr_positions': cdrs_global,
                                                'exposed_cdr': active, 'exposed_antigen': accessible_ag})
    settings = {'candidate': row['id'], 'profile': profile, 'seed': 42,
                'restraint_body_tool_default_seed': 917,
                'sampling': sampling, 'refine': refine, 'ncores': 6,
                'native_reference': native_reference and reference_source is None,
                'antibody_sha256': digest(antibody), 'antigen_sha256': digest(antigen),
                'ambig_sha256': digest(root / 'ambig.tbl'), 'unambig_sha256': digest(root / 'unambig.tbl')}
    config = f'''run_dir = "run"
mode = "local"
ncores = 6
debug = false
molecules = ["antibody.pdb", "antigen.pdb"]

[topoaa]

[rigidbody]
sampling = {sampling}
iniseed = 42
ambig_fname = "ambig.tbl"
unambig_fname = "unambig.tbl"

[seletop]
select = {refine}

[flexref]
iniseed = 42
ambig_fname = "ambig.tbl"
unambig_fname = "unambig.tbl"

[emref]
iniseed = 42
ambig_fname = "ambig.tbl"
unambig_fname = "unambig.tbl"

[clustfcc]
clust_cutoff = 0.6
min_population = 4

[seletopclusts]
top_clusters = 5
top_models = 4

[caprieval]
irmsd = false
fnat = false
lrmsd = false
ilrmsd = false
dockq = false
global_rmsd = false
'''
    if controlled:
        config = config.replace('[topoaa]\n', controlled_topology_block(controlled['histidines']))
        settings['controlled_policy'] = controlled
        settings['native_reference'] = controlled['antibody_source'] == 'experimental'
    (root / 'workflow.cfg').write_text(config, encoding='ascii')
    settings['config_sha256'] = digest(root / 'workflow.cfg')
    old = root / 'settings.json'
    if old.exists() and read_json(old) != settings:
        raise RuntimeError(f'Existing docking settings differ: {root}')
    write_json(old, settings)
    write_json(OUTPUT / 'docking' / profile / row['id'] / 'location.json',
               {'raw_results': str(root), 'settings': settings})
    return root


def matched_validation_structure(row, destination):
    """Match resolved residue IDs without borrowing any experimental coordinates."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    native = OUTPUT / 'references' / row['id'] / 'resolved_Fv_PDL1.pdb'
    fv, antigen = destination / 'predicted_Fv.pdb', destination / 'predicted_antigen.pdb'
    write_partner(predicted_structure(row['id']), {'H': row['VH_AA'], 'L': row['VL_AA']}, fv)
    write_partner(predicted_structure('PDL1_ECD'), {'A': read_json(OUTPUT / 'manifest.json')['antigen']}, antigen)
    def residues(lines):
        return {(line[21], line[22:27]): line[17:20] for line in lines if line.startswith('ATOM')}
    expected = residues(native.read_text().splitlines())
    predicted_lines = fv.read_text().splitlines() + antigen.read_text().splitlines()
    observed = residues(predicted_lines)
    missing = set(expected) - set(observed)
    if missing:
        raise ValueError(f'Missing matched predicted residues: {sorted(missing)}')
    kept = [line for line in predicted_lines if line.startswith('ATOM') and (line[21], line[22:27]) in expected]
    result = destination / 'predicted_matched_Fv_PDL1.pdb'
    result.write_text('\n'.join(kept) + '\nEND\n', encoding='ascii')
    write_json(destination / 'matched_input.json', {
        'candidate': row['id'], 'reference_sha256': digest(native),
        'predicted_fv_sha256': digest(fv), 'predicted_antigen_sha256': digest(antigen),
        'matched_sha256': digest(result), 'matched_residue_count': len(expected),
        'sequence_differences': [{'chain': k[0], 'residue': k[1].strip(),
                                  'experimental': expected[k], 'predicted': observed[k]}
                                 for k in sorted(expected) if expected[k] != observed[k]],
        'scope': 'Predicted coordinates only; same resolved residue IDs as experimental input. No template coordinates.'})
    return result


def capri_quality(model, reference, destination):
    """Official HADDOCK CAPRI; merge H/L into B to exclude internal H-L contacts."""
    import math
    import yaml
    import haddock.modules.analysis.caprieval as caprieval
    from haddock.libs.libcapri import CAPRI
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    def merged(source, name):
        source = Path(source)
        text = gzip.open(source, 'rt').read() if source.suffix == '.gz' else source.read_text()
        lines = []
        for line in text.splitlines():
            if line.startswith('ATOM'):
                chain = 'B' if line[21] in 'HLB' else line[21]
                if chain not in 'AB':
                    raise ValueError(f'Unexpected CAPRI chain {chain}')
                lines.append(line[:21] + chain + line[22:72].ljust(50) + chain.ljust(4) + line[76:])
        path = destination / name
        path.write_text('\n'.join(lines) + '\nEND\n', encoding='ascii')
        return path
    # B uses globally unique VH/VL residue numbers, not overlapping chain numbering.
    mod, ref = merged(model, 'model_AB.pdb'), merged(reference, 'reference_AB.pdb')
    defaults = yaml.safe_load((Path(caprieval.__file__).parent / 'defaults.yaml').read_text())
    params = {key: value['default'] for key, value in defaults.items()}
    params.update(receptor_chain='A', ligand_chains=['B'], global_rmsd=False, ilrmsd=False)
    value = CAPRI(1, mod, destination, ref, params).run()
    if value is None:
        raise ValueError('CAPRI sequence alignment failed')
    metrics = {key: float(getattr(value, key)) for key in ('fnat', 'irmsd', 'lrmsd', 'dockq')}
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError(f'Nonfinite CAPRI metrics: {metrics}')
    return metrics


def selected_control_models(root):
    """Cluster selection renames files; ori_name retains the minimized pose identity."""
    from haddock.libs.libontology import ModuleIO
    io = ModuleIO(); io.load(Path(root) / 'run/6_seletopclusts/io.json')
    models = [model for model in io.retrieve_models() if model.clt_rank == 1]
    models.sort(key=lambda model: model.clt_model_rank)
    return [str(model.ori_name) for model in models[:4]]


def evaluate_control_poses(row, profile, condition):
    """Evaluate every minimized pose, separating score ranking from oracle quality."""
    import inspect
    from haddock.libs.libontology import ModuleIO
    root = ENV / 'runs' / profile / row['id']
    reference = OUTPUT / 'references' / row['id'] / 'resolved_Fv_PDL1.pdb'
    directory = OUTPUT / 'control_validation' / condition / row['id']
    directory.mkdir(parents=True, exist_ok=True)
    io_path = root / 'run/4_emref/io.json'
    signature = {'version': 1, 'settings_sha256': digest(root / 'settings.json'),
                 'io_sha256': digest(io_path), 'reference_sha256': digest(reference),
                 'evaluator_sha256': hashlib.sha256(inspect.getsource(capri_quality).encode()).hexdigest()}
    io = ModuleIO(); io.load(io_path)
    models = sorted(io.retrieve_models(), key=lambda model: float(model.score))
    if len(models) != 100:
        raise ValueError(f'Expected 100 minimized models; found {len(models)}')
    def source_path(model):
        path = root / 'run/4_emref' / str(model.file_name)
        return path if path.exists() else path.with_suffix(path.suffix + '.gz')
    aggregate_signature = signature | {
        'pose_sha256': [digest(source_path(model)) for model in models],
        'selected_io_sha256': digest(root / 'run/6_seletopclusts/io.json'),
        'restoration_sha256': hashlib.sha256(inspect.getsource(restore_pose_chains).encode()).hexdigest(),
        'selection_mapper_sha256': hashlib.sha256(inspect.getsource(selected_control_models).encode()).hexdigest(),
        'VH_length': len(row['VH_AA'])}
    cached = directory / 'quality.json'
    if cached.exists() and read_json(cached)['signature'] == aggregate_signature:
        return read_json(cached)
    poses = []
    for rank, model in enumerate(models, 1):
        source = source_path(model)
        work = directory / f'pose_{rank:03d}'
        work.mkdir(exist_ok=True)
        restored = work / 'model_HL.pdb'
        restore_pose_chains(source, restored, len(row['VH_AA']))
        pose_signature = signature | {'source_sha256': digest(source)}
        pose_cache = work / 'quality.json'
        if pose_cache.exists() and read_json(pose_cache)['signature'] == pose_signature:
            quality = read_json(pose_cache)['metrics']
        else:
            quality = capri_quality(restored, reference, work)
            write_json(pose_cache, {'signature': pose_signature, 'metrics': quality})
        poses.append({'score_rank': rank, 'model': str(model.file_name), 'score': float(model.score), **quality})
    selected_names = set(selected_control_models(root))
    selected_poses = sorted((p for p in poses if p['model'] in selected_names), key=lambda p: p['score'])[:4]
    result = {'id': row['id'], 'condition': condition, 'profile': profile,
              'signature': aggregate_signature, 'poses': poses, 'model_count': len(poses),
              'best_dockq_in_score_top': {str(k): max(p['dockq'] for p in poses[:k]) for k in (1, 5, 10, 100)},
              'selected_cluster_top4': selected_poses,
              'best_dockq_selected_cluster_top4': max((p['dockq'] for p in selected_poses), default=None),
              'best_quality_score_rank': max(poses, key=lambda p: p['dockq'])['score_rank'],
              'scope': 'Fnat uses 5A contacts, iRMSD uses 10A interface; internal H-L contacts excluded. Top100 best is oracle diagnostic, not prospective selection.'}
    write_json(cached, result)
    return result


def controlled_topology_block(histidines):
    """A fixed modeling policy, not a claim about measured physiological protonation."""
    lines = ['[topoaa]', 'autohis = false', 'hydrogen_build = "all"', 'iniseed = 42']
    for index, residues in enumerate(histidines, 1):
        ids = sorted(set(residues))
        lines += [f'[topoaa.mol{index}]', 'charged_nter = false', 'charged_cter = false',
                  'nhisd = 0', f'nhise = {len(ids)}']
        lines += [f'hise_{n} = {residue}' for n, residue in enumerate(ids, 1)]
    return '\n'.join(lines) + '\n'


def controlled_partner_structure(fv_source, antigen_source, destination):
    """Keep Fv coordinates, use fixed experimental antigen; discard H/D and OXT."""
    from Bio.PDB import PDBParser, PDBIO, Structure, Model
    parser = PDBParser(QUIET=True)
    fv = parser.get_structure('fv', fv_source)[0]
    antigen = parser.get_structure('antigen', antigen_source)[0]
    result = Structure.Structure('controlled'); result.add(Model.Model(0))
    for chain, model in [('H', fv), ('L', fv), ('A', antigen)]:
        copied = model[chain].copy()
        for residue in copied:
            for atom in list(residue):
                if atom.element in {'H', 'D'} or atom.name == 'OXT': residue.detach_child(atom.id)
        result[0].add(copied)
    writer = PDBIO(); writer.set_structure(result); writer.save(str(destination))


def read_controlled_topology(folder, heavy_length):
    import io
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    atoms, bonds, hashes = {}, set(), {}
    for path in sorted(Path(folder).glob('*.psf*')):
        if not (path.name.endswith('.psf') or path.name.endswith('.psf.gz')): continue
        text = gzip.open(path, 'rt').read() if path.suffix == '.gz' else path.read_text()
        data = MMCIF2Dict(io.StringIO(text)); id_map = {}
        fields = ['id', 'segment_id', 'residue_id', 'residue_name', 'atom_name', 'chemical_type', 'charge', 'atom_mass']
        for atom_id, segment, resid, resname, name, chemical, charge, mass in zip(*(data['_cns_mtf_atom.'+f] for f in fields)):
            if atom_id == '-1': continue
            number = int(resid)
            chain = ('H' if number <= heavy_length else 'L') if segment == 'B' else segment
            key = f'{chain}:{number}:{name}'
            if key in atoms: raise ValueError(f'Duplicate processed atom: {key}')
            atoms[key] = {'resname': resname, 'type': chemical, 'charge': float(charge), 'mass': float(mass)}
            id_map[atom_id] = key
        for first, second in zip(data['_cns_mtf_bond.id[1]'], data['_cns_mtf_bond.id[2]']):
            if first == second == '-1': continue
            bonds.add(tuple(sorted((id_map[first], id_map[second]))))
        hashes[path.name] = digest(path)
    if not atoms or not bonds: raise ValueError('Processed topology atoms/bonds missing')
    return {'atoms': atoms, 'bonds': sorted([list(edge) for edge in bonds]), 'PSF_sha256': hashes}


def compare_controlled_chemistry(first, second):
    keys = set(first['atoms']) | set(second['atoms'])
    changed = [key for key in sorted(keys) if first['atoms'].get(key) != second['atoms'].get(key)]
    bonds1, bonds2 = {tuple(e) for e in first['bonds']}, {tuple(e) for e in second['bonds']}
    structural_cis = []
    for key in changed:
        one, two = first['atoms'].get(key), second['atoms'].get(key)
        if one and two and key.endswith(':C') and {one.get('type'), two.get('type')} == {'C', 'CCIS'}:
            if {k: v for k, v in one.items() if k != 'type'} == {k: v for k, v in two.items() if k != 'type'}:
                structural_cis.append(key)
    return {'identical': not changed and bonds1 == bonds2, 'atom_differences': changed,
            'protonation_charge_connectivity_identical': set(changed) == set(structural_cis) and bonds1 == bonds2,
            'structural_cis_type_differences': structural_cis,
            'experimental_only_bonds': sorted(bonds1 - bonds2), 'predicted_only_bonds': sorted(bonds2 - bonds1),
            'experimental_atoms': len(first['atoms']), 'predicted_atoms': len(second['atoms']),
            'experimental_charge_e': sum(a['charge'] for a in first['atoms'].values()),
            'predicted_charge_e': sum(a['charge'] for a in second['atoms'].values())}


def prepare_controlled_controls():
    """Prepare four matched conditions and run topology only; never dock here."""
    from Bio.PDB import PDBParser, PDBIO, Select
    from Bio.SeqUtils import seq1
    from haddock.clis.restraints.calc_accessibility import get_accessibility, apply_cutoff
    directory = OUTPUT / 'controlled_controls_20261120'; directory.mkdir(exist_ok=True)
    state = {'status': 'preparing', 'seed': 42, 'ncores': 6, 'sampling': 1000, 'refine': 100,
             'docking_started': False, 'pairs': [], 'runner_sha256': digest(Path(__file__)),
             'policy': 'Per-control fixed cognate experimental antigen; same observed sequence/mapped residues/AIR; all histidines neutral HISE; uncharged termini. Input Fv conformations and their internal-distance/cis-specific typing consequences are preserved and reported. Not measured physiological protonation.'}
    write_json(directory / 'state.json', state)
    parser = PDBParser(QUIET=True)
    for row in read_json(OUTPUT / 'manifest.json')['candidates']:
        if row['group'] != 'control': continue
        native = OUTPUT / 'references' / row['id'] / 'resolved_Fv_PDL1.pdb'
        predicted = OUTPUT / 'control_validation/inputs' / row['id'] / 'predicted_matched_Fv_PDL1.pdb'
        inputs = directory / row['id']; inputs.mkdir(exist_ok=True)
        sources = {}
        for kind, fv in [('experimental', native), ('predicted', predicted)]:
            path = inputs / f'{kind}_Fv_fixed_antigen.pdb'
            controlled_partner_structure(fv, native, path); sources[kind] = path
        models = {kind: parser.get_structure(kind, path)[0] for kind, path in sources.items()}
        sequence_maps = {kind: {chain.id: [(r.id[1], seq1(r.resname)) for r in chain if r.id[0] == ' ']
                               for chain in model} for kind, model in models.items()}
        if sequence_maps['experimental'] != sequence_maps['predicted']:
            raise ValueError('Controlled inputs do not have identical residue identities/numbering')
        missing_residues = {}
        for chain, key in [('H', 'VH_AA'), ('L', 'VL_AA')]:
            offset = len(row['VH_AA']) if chain == 'L' else 0
            observed = {index - offset: aa for index, aa in sequence_maps['experimental'][chain]}
            if any(not 1 <= index <= len(row[key]) or row[key][index - 1] != aa for index, aa in observed.items()):
                raise ValueError('Controlled antibody residue identity differs from manifest')
            missing_residues[chain] = sorted(set(range(1, len(row[key]) + 1)) - set(observed))
        # Exact coordinate/identity equality of the common antigen, not just sequence equality.
        def antigen_atoms(model):
            return [(r.id, a.name, tuple(float(v) for v in a.coord)) for r in model['A'] for a in r]
        if antigen_atoms(models['experimental']) != antigen_atoms(models['predicted']):
            raise ValueError('Antigen coordinates differ between controlled inputs')
        class Chains(Select):
            def __init__(self, names): self.names = names
            def accept_chain(self, chain): return chain.id in self.names
        access = {}
        for kind, model in models.items():
            writer = PDBIO(); writer.set_structure(model)
            fv_path, ag_path = inputs / f'{kind}_Fv_isolated.pdb', inputs / f'{kind}_antigen_isolated.pdb'
            writer.save(str(fv_path), Chains({'H', 'L'})); writer.save(str(ag_path), Chains({'A'}))
            # FreeSASA's C API rejects non-ASCII paths; keep a short-lived ASCII copy.
            import tempfile
            with tempfile.TemporaryDirectory(prefix='pdl1-controlled-access-') as temporary:
                temp_fv, temp_ag = Path(temporary) / 'fv.pdb', Path(temporary) / 'antigen.pdb'
                shutil.copy2(fv_path, temp_fv); shutil.copy2(ag_path, temp_ag)
                access[kind] = apply_cutoff(get_accessibility(str(temp_fv)), .4)
                access[kind]['A'] = apply_cutoff(get_accessibility(str(temp_ag)), .4)['A']
        cdr = cdr_positions(row); fixed = {}
        for chain in ('H', 'L'):
            positions = {i + (len(row['VH_AA']) if chain == 'L' else 0)
                         for name, ids in cdr.items() if name.endswith('.' + chain) for i in ids}
            fixed[chain] = sorted(positions & set(access['experimental'][chain]) & set(access['predicted'][chain]))
        fixed['A'] = sorted(set(access['experimental']['A']) & set(access['predicted']['A']))
        if not all(fixed.values()): raise ValueError('No shared exposed restraint residues')
        histidines = [[r.id[1] for chain in ('H', 'L') for r in models['experimental'][chain] if r.resname == 'HIS'],
                      [r.id[1] for r in models['experimental']['A'] if r.resname == 'HIS']]
        roots, chemistry = {}, {}
        for kind in ('experimental', 'predicted'):
            profile = 'reference-pilot-controlled-20261120-' + kind
            policy = {'antibody_source': kind, 'fixed_AIR': fixed, 'histidines': histidines,
                      'histidine_policy': 'all neutral HISE, autohis disabled',
                      'antigen_policy': 'same cognate experimental antigen for both Fv inputs',
                      'internal_restraints': 'same restrain_bodies algorithm/default seed917; input-dependent target distances retained, not frozen to experimental geometry'}
            root = prepare_docking(row, 1000, 100, profile, True, reference_source=sources[kind], controlled=policy)
            roots[kind] = root
            check = (root / 'workflow.cfg').read_text().split('[rigidbody]')[0].replace('run_dir = "run"', 'run_dir = "topology_check"')
            cfg = root / 'topology_check.cfg'; cfg.write_text(check, encoding='ascii')
            marker = root / 'topology_check_ready.json'
            signature = {'config_sha256': digest(cfg), 'settings_sha256': digest(root / 'settings.json')}
            if marker.exists():
                if read_json(marker)['signature'] != signature: raise RuntimeError('Topology preflight settings changed')
            else:
                if (root / 'topology_check').exists(): raise RuntimeError(f'Interrupted topology preflight: {root}')
                run_logged([ENV / 'docking/bin/haddock3', cfg.name], root / 'topology_check.log', cwd=root,
                           extra_env={'PATH': str(ENV / 'docking/bin') + ':' + os.environ.get('PATH', '')})
            chemistry[kind] = read_controlled_topology(root / 'topology_check/0_topoaa', len(row['VH_AA']))
            histidine_checks = []
            for molecule, ids in enumerate(histidines):
                for number in ids:
                    chain = ('H' if number <= len(row['VH_AA']) else 'L') if molecule == 0 else 'A'
                    prefix = f'{chain}:{number}:'
                    atoms = {key.rsplit(':', 1)[-1]: value for key, value in chemistry[kind]['atoms'].items() if key.startswith(prefix)}
                    charge = sum(value['charge'] for value in atoms.values())
                    valid = 'HE2' in atoms and 'HD1' not in atoms and abs(charge) < 1e-5
                    if not valid: raise ValueError(f'Processed histidine is not neutral HISE: {prefix}')
                    histidine_checks.append({'residue': f'{chain}:{number}', 'HISE_verified': True, 'charge_e': charge})
            chemistry[kind]['histidine_policy_checks'] = histidine_checks
            write_json(root / 'topology_chemistry.json', chemistry[kind])
            if marker.exists() and read_json(marker)['PSF_sha256'] != chemistry[kind]['PSF_sha256']:
                raise RuntimeError('Cached topology PSF files changed')
            write_json(marker, {'signature': signature, 'PSF_sha256': chemistry[kind]['PSF_sha256']})
        report = compare_controlled_chemistry(chemistry['experimental'], chemistry['predicted'])
        from Bio.PDB import calc_dihedral
        import math
        cis_geometry = []
        for key in report['structural_cis_type_differences']:
            chain, residue, _ = key.split(':'); residue = int(residue); angles = {}
            for kind, model in models.items():
                first, second = model[chain][residue], model[chain][residue + 1]
                omega = math.degrees(calc_dihedral(first['CA'].get_vector(), first['C'].get_vector(),
                                                   second['N'].get_vector(), second['CA'].get_vector()))
                if (chemistry[kind]['atoms'][key]['type'] == 'CCIS') != (abs(omega) < 90):
                    raise ValueError('Cis-specific atom typing is inconsistent with measured input geometry')
                angles[kind] = float(omega)
            cis_geometry.append({'atom': key, 'omega_degrees': angles,
                                 'scope': 'Input cis/trans structural difference preserved; charges/masses/connectivity equal, force-field C/CCIS types differ.'})
        report.update({'id': row['id'], 'fixed_AIR': fixed, 'histidines': histidines,
                       'preserved_structural_cis_differences': cis_geometry,
                       'resolved_sequence_maps': sequence_maps['experimental'],
                       'source_sha256': {'experimental_complex': digest(native), 'predicted_matched_complex': digest(predicted)},
                       'sequence_and_residue_mapping_identical': True,
                       'unresolved_antibody_sequence_positions_omitted_in_both': missing_residues,
                       'antigen_PDB_identical': digest(roots['experimental'] / 'antigen.pdb') == digest(roots['predicted'] / 'antigen.pdb'),
                       'AIR_table_identical': digest(roots['experimental'] / 'ambig.tbl') == digest(roots['predicted'] / 'ambig.tbl'),
                       'topology_policy_identical': (roots['experimental'] / 'topology_check.cfg').read_text() == (roots['predicted'] / 'topology_check.cfg').read_text(),
                       'internal_Fv_distance_targets_identical': digest(roots['experimental'] / 'unambig.tbl') == digest(roots['predicted'] / 'unambig.tbl'),
                       'root_paths': {k: str(v) for k, v in roots.items()},
                       'sequence_scope': 'Same within each antibody control pair; two controls retain their own experimental PD-L1 conformations/resolved scopes.'})
        write_json(inputs / 'normalization_report.json', report)
        if not all(report[k] for k in ('protonation_charge_connectivity_identical', 'antigen_PDB_identical', 'AIR_table_identical', 'topology_policy_identical')):
            state.update(status='failed', failed_pair=report); write_json(directory / 'state.json', state)
            raise RuntimeError('Controlled sequence/antigen/AIR/chemistry matching failed')
        state['pairs'].append(report); write_json(directory / 'state.json', state)
    state.update(status='ready', docking_started=False)
    write_json(directory / 'state.json', state)
    print('Four controlled inputs READY: identical per-pair sequence/antigen/AIR/protonation/charges/bonds. Input cis-specific atom types are separately reported. Topology only, no docking.', flush=True)


def control_selection_orders(poses, cluster_order, cluster_representatives):
    """Fixed prospective orders: never inspect DockQ/native contacts to select a pose."""
    global_order = [p['model'] for p in sorted(poses, key=lambda p: p['score_rank'])]
    hybrid = []
    for index in range(5):
        for source in (global_order[:5], cluster_representatives[:5]):
            if index < len(source) and source[index] not in hybrid:
                hybrid.append(source[index])
    return {'cluster_first': list(cluster_order), 'global_score': global_order, 'diversity_rescue': hybrid}


def fit_control_coordinates(predicted, reference):
    """Proper Kabsch rotation for row vectors; no scaling or reflection."""
    import numpy as np
    predicted, reference = np.asarray(predicted, dtype=float), np.asarray(reference, dtype=float)
    if predicted.shape != reference.shape or predicted.ndim != 2 or predicted.shape[1] != 3 or len(predicted) < 3:
        raise ValueError('At least three paired 3D coordinates are required')
    u, _, vt = np.linalg.svd((predicted - predicted.mean(0)).T @ (reference - reference.mean(0)))
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    rotation = u @ vt
    return rotation, reference.mean(0) - predicted.mean(0) @ rotation


def audit_control_structure(row, destination):
    import numpy as np
    import io
    from Bio.PDB import PDBParser, PDBIO, Select
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    source = OUTPUT / 'control_validation'
    native_file = OUTPUT / 'references' / row['id'] / 'resolved_Fv_PDL1.pdb'
    predicted_file = source / 'inputs' / row['id'] / 'predicted_matched_Fv_PDL1.pdb'
    parser = PDBParser(QUIET=True)
    native = parser.get_structure('native', native_file)[0]
    predicted = parser.get_structure('predicted', predicted_file)[0]
    def coordinates(model):
        return {(c.id, r.id[1], a.name): np.array(a.coord, dtype=float)
                for c in model for r in c if r.id[0] == ' ' for a in r if a.element not in {'H', 'D'}}
    ref, pred = coordinates(native), coordinates(predicted)
    common = sorted(set(ref) & set(pred)); backbone = {'N', 'CA', 'C', 'O'}
    cdr = cdr_positions(row)
    cdr = {name: {i + (len(row['VH_AA']) if name.endswith('.L') else 0) for i in ids}
           for name, ids in cdr.items()}
    def fit(keys):
        return fit_control_coordinates([pred[k] for k in keys], [ref[k] for k in keys])
    def rmsd(keys, rotation, translation):
        return float(np.sqrt(np.mean([np.sum((pred[k] @ rotation + translation - ref[k])**2) for k in keys]))) if keys else None
    fits, metrics = {}, []
    for chain in ('H', 'L', 'A'):
        keys = [k for k in common if k[0] == chain and k[2] in backbone]
        rotation, translation = fit(keys)
        metrics.append({'region': chain + '_backbone_self_fit', 'n_atoms': len(keys), 'rmsd_A': rmsd(keys, rotation, translation)})
        if chain in 'HL':
            excluded = set().union(*(ids for name, ids in cdr.items() if name.endswith('.' + chain)))
            framework = [k for k in keys if k[1] not in excluded]
            fr_rot, fr_trans = fit(framework)
            fits[chain] = (fr_rot, fr_trans)
            metrics.append({'region': chain + '_framework_self_fit', 'n_atoms': len(framework),
                            'rmsd_A': rmsd(framework, fr_rot, fr_trans)})
            for name, ids in cdr.items():
                if name.endswith('.' + chain):
                    region = [k for k in keys if k[1] in ids]
                    metrics.append({'region': name + '_after_framework_fit', 'n_atoms': len(region),
                                    'expected_residues': len(ids), 'observed_residues': len({k[1] for k in region}),
                                    'rmsd_A': rmsd(region, fr_rot, fr_trans)})
        else:
            fits['A'] = (rotation, translation)
    h_rot, h_trans = fits['H']
    light = [k for k in common if k[0] == 'L' and k[2] in backbone]
    metrics.append({'region': 'VL_backbone_after_VH_framework_fit', 'n_atoms': len(light),
                    'rmsd_A': rmsd(light, h_rot, h_trans)})
    reference_metrics = read_json(OUTPUT / 'references' / row['id'] / 'reference_metrics.json')
    interface_residues = {p[2] for p in reference_metrics['interface']['contact_pairs']}
    a_rot, a_trans = fits['A']
    interface_bb = [k for k in common if k[0] == 'A' and k[1] in interface_residues and k[2] in backbone]
    same_residue = {r.id[1] for r in native['A'] if r.id in predicted['A'] and r.resname == predicted['A'][r.id].resname}
    interface_sc = [k for k in common if k[0] == 'A' and k[1] in interface_residues & same_residue and k[2] not in backbone]
    for name, keys in [('antigen_interface_backbone_after_A_fit', interface_bb),
                       ('antigen_interface_sidechain_identical_residues_after_A_fit', interface_sc)]:
        metrics.append({'region': name, 'n_atoms': len(keys), 'rmsd_A': rmsd(keys, a_rot, a_trans)})
    per_residue = []
    for key in common:
        if key[2] == 'CA':
            rotation, translation = fits['A'] if key[0] == 'A' else fits['H']
            per_residue.append({'chain': key[0], 'residue': key[1],
                                'CA_displacement_A': float(np.linalg.norm(pred[key] @ rotation + translation - ref[key])),
                                'predicted_pLDDT': float(predicted[key[0]][key[1]]['CA'].bfactor)})
    conditions, processed_atoms = {}, {}
    for kind in ('experimental', 'predicted_matched'):
        root = ENV / 'runs' / ('reference-pilot-validation-20261119-' + kind) / row['id']
        maps = read_json(root / 'residue_map.json')
        atom_counts = {'heavy': 0, 'hydrogen': 0}
        for path in (root / 'run/0_topoaa').glob('*.pdb*'):
            if not (path.name.endswith('.pdb') or path.name.endswith('.pdb.gz')):
                continue
            text = gzip.open(path, 'rt').read() if path.suffix == '.gz' else path.read_text()
            model = parser.get_structure('topology', io.StringIO(text))[0]
            for atom in model.get_atoms():
                atom_counts['hydrogen' if atom.element in {'H', 'D'} else 'heavy'] += 1
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        atoms, psf_hashes = {}, {}
        for path in (root / 'run/0_topoaa').glob('*.psf*'):
            if not (path.name.endswith('.psf') or path.name.endswith('.psf.gz')):
                continue
            text = gzip.open(path, 'rt').read() if path.suffix == '.gz' else path.read_text()
            data = MMCIF2Dict(io.StringIO(text))
            columns = ['id', 'segment_id', 'residue_id', 'residue_name', 'atom_name', 'chemical_type', 'charge']
            for atom_id, segment, resid, resname, name, chemical, charge in zip(*(data['_cns_mtf_atom.' + key] for key in columns)):
                if atom_id == '-1':  # CNS loop terminator, not an atom
                    continue
                index = int(resid)
                chain = ('H' if index <= len(row['VH_AA']) else 'L') if segment == 'B' else segment
                key = f'{chain}:{index}:{name}'
                if key in atoms: raise ValueError(f'Duplicate processed atom: {key}')
                atoms[key] = {'chain': chain, 'residue': index, 'resname': resname, 'atom': name,
                              'chemical_type': chemical, 'charge_e': float(charge)}
            psf_hashes[path.name] = digest(path)
        if len(atoms) != sum(atom_counts.values()):
            raise ValueError('Processed PSF and PDB atom counts differ')
        processed_atoms[kind] = atoms
        conditions[kind] = {'exposed_cdr': maps['exposed_cdr'], 'exposed_antigen': maps['exposed_antigen'],
                            'post_topology_atom_counts': atom_counts, 'processed_PSF_sha256': psf_hashes,
                            'total_topology_charge_e': sum(v['charge_e'] for v in atoms.values()),
                            'settings_sha256': digest(root / 'settings.json')}
    post_differences = [{'key': key, 'experimental': processed_atoms['experimental'].get(key),
                         'predicted_matched': processed_atoms['predicted_matched'].get(key)}
                        for key in sorted(set(processed_atoms['experimental']) | set(processed_atoms['predicted_matched']))
                        if processed_atoms['experimental'].get(key) != processed_atoms['predicted_matched'].get(key)]
    with (destination / 'processed_atoms.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['condition', 'chain', 'residue', 'resname', 'atom', 'chemical_type', 'charge_e'])
        writer.writeheader()
        for kind, atoms in processed_atoms.items():
            for atom in atoms.values(): writer.writerow({'condition': kind, **atom})
    air = {}
    for chain in ('H', 'L', 'A'):
        def exposed(kind):
            return set(conditions[kind]['exposed_antigen'] if chain == 'A' else conditions[kind]['exposed_cdr'][chain])
        first, second = exposed('experimental'), exposed('predicted_matched')
        air[chain] = {'experimental_n': len(first), 'predicted_n': len(second),
                      'common': sorted(first & second), 'experimental_only': sorted(first - second),
                      'predicted_only': sorted(second - first),
                      'jaccard': len(first & second) / len(first | second) if first | second else None}
    confidence = next(r for r in read_json(OUTPUT / 'folding_summary.json')['results'] if r['id'] == row['id'])
    result = {'id': row['id'], 'metrics': metrics, 'per_residue': per_residue,
              'native_sha256': digest(native_file), 'predicted_sha256': digest(predicted_file),
              'sequence_differences': read_json(source / 'inputs' / row['id'] / 'matched_input.json')['sequence_differences'],
              'native_only_heavy_atom_keys': [list(k) for k in sorted(set(ref) - set(pred))],
              'predicted_only_heavy_atom_keys': [list(k) for k in sorted(set(pred) - set(ref))],
              'native_resolved_residues': {c.id: len(c) for c in native},
              'predicted_resolved_residues': {c.id: len(c) for c in predicted},
              'conditions': conditions, 'post_topology_atom_differences': post_differences,
              'AIR_comparison': air, 'Fv_confidence': confidence,
              'scope': 'Observed common backbone atoms. Each chain fitted independently except VL-after-VH. Interface defined from native 4A contacts for audit only. Side-chain RMSD excludes sequence mismatches and is not symmetry corrected. No causal attribution or binding prediction.'}
    class Chains(Select):
        def __init__(self, names): self.names = names
        def accept_chain(self, chain): return chain.id in self.names
    for label, chain_names, rotation, translation in [('Fv', {'H', 'L'}, h_rot, h_trans), ('antigen', {'A'}, a_rot, a_trans)]:
        writer = PDBIO(); writer.set_structure(native)
        writer.save(str(destination / f'native_{label}.pdb'), Chains(chain_names))
        aligned = predicted.copy()
        for atom in aligned.get_atoms(): atom.transform(rotation, translation)
        writer.set_structure(aligned); writer.save(str(destination / f'predicted_{label}_aligned.pdb'), Chains(chain_names))
    (destination / 'view.pml').write_text('''reinitialize
load native_Fv.pdb, native_Fv
load predicted_Fv_aligned.pdb, predicted_Fv
load native_antigen.pdb, native_antigen
load predicted_antigen_aligned.pdb, predicted_antigen
hide everything
show cartoon
color gray70, native_Fv or native_antigen
color teal, predicted_Fv
color orange, predicted_antigen
disable all
enable native_antigen
enable predicted_antigen
orient native_antigen
scene antigen_independent_fit, store
disable all
enable native_Fv
enable predicted_Fv
orient native_Fv
scene Fv_VH_framework_fit, store
''', encoding='ascii')
    write_json(destination / 'structure_audit.json', result)
    return result


def audit_controls():
    """Steps 1+2 only: saved pose selection audit and component structural alignment."""
    from collections import Counter
    from haddock.libs.libontology import ModuleIO
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    source = OUTPUT / 'control_validation'
    directory = OUTPUT / 'control_audit_20261120'; directory.mkdir(exist_ok=True)
    results = read_json(source / 'comparison.json')
    traces, fixed, budgets, audit_results = [], [], [], []
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for index, result in enumerate(results):
        root = ENV / 'runs' / result['profile'] / result['id']
        with (root / 'run/5_clustfcc/clustfcc.tsv').open() as stream:
            clustered = {Path(r['model_name']).name: r['cluster_id'] for r in csv.DictReader(stream, delimiter='\t')}
        sizes = Counter(v for v in clustered.values() if v not in {'-', '', 'nan'})
        io = ModuleIO(); io.load(root / 'run/6_seletopclusts/io.json')
        selected = sorted(io.retrieve_models(), key=lambda v: (v.clt_rank, v.clt_model_rank))
        cluster_order = [str(v.ori_name) for v in selected]
        representatives = [str(v.ori_name) for v in selected if v.clt_model_rank == 1]
        orders = control_selection_orders(result['poses'], cluster_order, representatives)
        by_name = {p['model']: p for p in result['poses']}
        original = set(selected_control_models(root))
        good = {p['model'] for p in result['poses'] if p['dockq'] >= .23}
        for pose in result['poses']:
            cluster = clustered[pose['model']]
            traces.append({'id': result['id'], 'condition': result['condition'], **pose,
                           'cluster_id': cluster, 'cluster_size': sizes.get(cluster, 0) if cluster != '-' else None,
                           'selected_original_top4': pose['model'] in original,
                           'selected_global_top5': pose['model'] in orders['global_score'][:5],
                           'selected_diversity_rescue': pose['model'] in orders['diversity_rescue']})
        def measure(names):
            return {'selected_n': len(names), 'best_dockq': max((by_name[n]['dockq'] for n in names), default=None),
                    'available_acceptable_n': len(good), 'acceptable_selected_n': len(set(names) & good),
                    'acceptable_pose_recall': len(set(names) & good) / len(good) if good else None,
                    'any_acceptable': bool(set(names) & good)}
        for rule, names in [('original_cluster_top4', list(original)), ('global_top5', orders['global_score'][:5]),
                            ('diversity_rescue', orders['diversity_rescue'])]:
            fixed.append({'id': result['id'], 'condition': result['condition'], 'rule': rule, **measure(names)})
        for rule, order in orders.items():
            for k in range(1, 11):
                if len(order) >= k:  # exact equal budgets, no silent padding/fewer candidates
                    budgets.append({'id': result['id'], 'condition': result['condition'], 'rule': rule, 'K': k, **measure(order[:k])})
        import tomllib
        from haddock.modules.analysis.clustfcc import create_elements
        parameters = tomllib.loads((root / 'run/5_clustfcc/params.cfg').read_text())['clustfcc']
        matrix_path = root / 'run/5_clustfcc/fcc.matrix'
        matrix = [(int(v[0]), int(v[1]), float(v[2]), float(v[3]))
                  for v in (line.split() for line in matrix_path.read_text().splitlines())]
        pool = create_elements(matrix, parameters['clust_cutoff'], parameters['strictness'])
        previous = ModuleIO(); previous.load(root / 'run/4_emref/io.json')
        input_models = previous.retrieve_models(individualize=True)
        names = [str(model.file_name) for model in input_models]
        leader = orders['global_score'][0]
        element = pool[names.index(leader) + 1]
        neighbor_names = sorted(names[n.name - 1] for n in element.neighbors)
        audit_results.append({'id': result['id'], 'condition': result['condition'], 'orders': orders,
                              'unclustered_n': sum(v == '-' for v in clustered.values()),
                              'score_leader_FCC_diagnostic': {'model': leader, 'neighbors': neighbor_names,
                                  'neighbor_count_with_self': len(neighbor_names) + 1,
                                  'min_population': parameters['min_population'],
                                  'cutoff': parameters['clust_cutoff'], 'strictness': parameters['strictness'],
                                  'scope': 'Adjacency from stored rounded FCC matrix at original parameters; not reclustering or a parameter sweep.'},
                              'FCC_matrix_sha256': digest(matrix_path),
                              'cluster_sizes': dict(sizes), 'cluster_table_sha256': digest(root / 'run/5_clustfcc/clustfcc.tsv'),
                              'selected_io_sha256': digest(root / 'run/6_seletopclusts/io.json')})
        axis = axes.flat[index]; axis2 = axes2.flat[index]
        for p in result['poses']:
            c = clustered[p['model']]
            axis.scatter(p['score'], p['dockq'], s=22, marker='x' if c == '-' else 'o',
                         color='#969696' if c == '-' else plt.get_cmap('tab20')(int(c) % 20), alpha=.8)
            if p['model'] in original:
                axis.scatter(p['score'], p['dockq'], s=85, facecolors='none', edgecolors='black')
        title = result['id'].removeprefix('CTRL_') + ' | ' + result['condition']
        axis.set(title=title, xlabel='HADDOCK score (lower is better)', ylabel='DockQ', ylim=(-.02, 1.02))
        axis.axhline(.23, color='#aaaaaa', linestyle='--', linewidth=.7); axis.grid(alpha=.15)
        for rule, color in [('cluster_first', '#657b98'), ('global_score', '#168373'), ('diversity_rescue', '#db872b')]:
            points = [p for p in budgets if p['id'] == result['id'] and p['condition'] == result['condition'] and p['rule'] == rule]
            axis2.plot([p['K'] for p in points], [p['best_dockq'] for p in points], marker='.', label=rule, color=color)
        axis2.axhline(max(p['dockq'] for p in result['poses']), color='black', linestyle=':', linewidth=.8, label='oracle all100')
        axis2.set(title=title, xlabel='Exactly K retained poses', ylabel='Best DockQ', ylim=(-.02, 1.02), xticks=[1, 2, 4, 5, 10])
        axis2.grid(alpha=.15)
    axes.flat[0].legend(handles=[Line2D([], [], marker='x', linestyle='', color='gray', label='unclustered'),
                                Line2D([], [], marker='o', linestyle='', color='black', markerfacecolor='none', label='original selected top4')], fontsize=8)
    axes2.flat[0].legend(fontsize=7)
    fig.suptitle('Selection audit: cluster membership and discarded high-quality poses')
    fig2.suptitle('Equal pose budgets: quality is evaluated after selection\nTwo antibodies, six conditions; descriptive development analysis')
    for figure, name in [(fig, 'score_quality_clusters'), (fig2, 'equal_budget_selection')]:
        for extension in ('png', 'svg'): figure.savefig(directory / f'{name}.{extension}', dpi=180)
        plt.close(figure)
    structures = []
    for row in read_json(OUTPUT / 'manifest.json')['candidates']:
        if row['group'] == 'control': structures.append(audit_control_structure(row, directory / 'structures' / row['id']))
    flat_structure = [{'id': r['id'], **v} for r in structures for v in r['metrics']]
    def table(name, records):
        fields = list(dict.fromkeys(k for r in records for k in r))
        with (directory / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    for name, records in [('pose_trace.csv', traces), ('selection_rules.csv', fixed), ('equal_budget.csv', budgets), ('structure_metrics.csv', flat_structure)]:
        table(name, records)
    fig3, ax3 = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig4 = plt.figure(figsize=(12, 9), constrained_layout=True)
    from Bio.PDB import PDBParser
    for index, result in enumerate(structures):
        folder = directory / 'structures' / result['id']
        for column, label in enumerate(('Fv', 'antigen')):
            axis = ax3[index, column]
            for chain in (('H', 'L') if label == 'Fv' else ('A',)):
                values = [r for r in result['per_residue'] if r['chain'] == chain]
                axis.plot([r['residue'] for r in values], [r['CA_displacement_A'] for r in values], label=chain)
            axis.set(title=result['id'].removeprefix('CTRL_') + ' | ' + label, xlabel='Mapped residue number', ylabel='CA displacement (A)')
            axis.legend(); axis.grid(alpha=.2)
            three = fig4.add_subplot(2, 2, index * 2 + column + 1, projection='3d')
            native = PDBParser(QUIET=True).get_structure('native', folder / f'native_{label}.pdb')[0]
            coords = np.array([a.coord for a in native.get_atoms() if a.name == 'CA'])
            center = coords.mean(0); radius = max(np.ptp(coords, axis=0)) / 2 + 2
            for kind, path, color in [('experimental', folder / f'native_{label}.pdb', '#777777'),
                                      ('predicted', folder / f'predicted_{label}_aligned.pdb', '#168373')]:
                model = PDBParser(QUIET=True).get_structure(kind, path)[0]
                for chain in model:
                    points = np.array([r['CA'].coord for r in chain if 'CA' in r])
                    three.plot(*points.T, color=color, alpha=.8, linewidth=1.2)
            three.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius), title=result['id'].removeprefix('CTRL_') + ' | ' + label)
            three.set_box_aspect((1, 1, 1)); three.set_axis_off()
    fig3.suptitle('Input deviation after alignment: Fv fitted on VH framework; antigen fitted independently')
    fig4.suptitle('Independent component overlays (CA traces)\nGray experimental, green predicted; not a predicted binding complex')
    for figure, name in [(fig3, 'input_CA_deviation'), (fig4, 'input_overlays')]:
        for extension in ('png', 'svg'): figure.savefig(directory / f'{name}.{extension}', dpi=180)
        plt.close(figure)
    write_json(directory / 'audit.json', {'status': 'completed', 'source_sha256': digest(source / 'comparison.json'),
               'runner_sha256': digest(Path(__file__)), 'conditions': audit_results, 'fixed_rules': fixed, 'equal_budgets': budgets,
               'structures': structures, 'pose_count': len(traces), 'dockq_acceptable_threshold': .23,
               'interpretation': 'No docking/inference rerun. Selection never uses native quality. cluster_first extends original top-cluster rule for K>4. No-acceptable-pose recall is null, not zero. Six conditions are not six independent antibodies; no significance claim. No min_population/cutoff tuning performed.'})
    print(f'Control audit complete: {len(traces)} saved poses, {len(structures)} input structure pairs; no CNS/inference.', flush=True)


def plot_control_validation(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'experimental': '#168373', 'predicted_matched': '#db872b', 'predicted_fullECD': '#657b98'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for axis, candidate in zip(axes, ('CTRL_atezolizumab', 'CTRL_durvalumab')):
        for result in results:
            if result['id'] != candidate:
                continue
            axis.plot((1, 5, 10, 100), [result['best_dockq_in_score_top'][str(k)] for k in (1, 5, 10, 100)],
                      marker='o', label=result['condition'], color=colors[result['condition']])
        axis.set(xscale='log', xticks=[1, 5, 10, 100], xticklabels=['1', '5', '10', '100'],
                 ylim=(0, 1.02), xlabel='Number of score-ranked poses inspected', ylabel='Best DockQ',
                 title=candidate.removeprefix('CTRL_'))
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.suptitle('Control validation: pose recovery versus score ranking\nExperimental / matched predicted / original full ECD inputs')
    folder = OUTPUT / 'control_validation'
    for suffix in ('png', 'svg'):
        fig.savefig(folder / f'control_validation.{suffix}', dpi=180)
    plt.close(fig)


def validate_controls():
    """Frozen four-run diagnostic; original main results and inputs are untouched."""
    controls = [r for r in read_json(OUTPUT / 'manifest.json')['candidates'] if r['group'] == 'control']
    directory = OUTPUT / 'control_validation'
    directory.mkdir(exist_ok=True)
    results = []
    tasks = [(kind, row) for kind in ('experimental', 'predicted_matched') for row in controls]
    state = {'status': 'running', 'seed': 42, 'ncores': 6, 'sampling': 1000, 'refine': 100,
             'task_count': 4, 'completed_tasks': [], 'main_results_preserved': True,
             'runner_sha256': digest(Path(__file__)),
             'scope': 'Bound-extracted redocking and residue-matched predicted docking; evaluation-only native reference, no native-epitope restraints. Single seed; not a generative generalization test.'}
    write_json(directory / 'state.json', state)
    try:
        for kind, row in tasks:
            profile = 'reference-pilot-validation-20261119-' + kind
            state.update(current_candidate=row['id'], current_condition=kind, current_profile=profile)
            write_json(directory / 'state.json', state)
            root = ENV / 'runs' / profile / row['id']
            if (root / 'run').exists() and not (root / 'completed.json').exists():
                raise RuntimeError(f'Interrupted validation: {root}; preserve outputs and inspect restart stage')
            source = matched_validation_structure(row, directory / 'inputs' / row['id']) if kind == 'predicted_matched' else None
            root = prepare_docking(row, 1000, 100, profile, True, reference_source=source)
            if not (root / 'completed.json').exists():
                run_logged([ENV / 'docking/bin/haddock3', 'workflow.cfg'], root / 'execution.log', cwd=root,
                           extra_env={'PATH': str(ENV / 'docking/bin') + ':' + os.environ.get('PATH', '')})
                write_json(root / 'completed.json', {'settings_sha256': digest(root / 'settings.json')})
            elif read_json(root / 'completed.json')['settings_sha256'] != digest(root / 'settings.json'):
                raise RuntimeError(f'Completed validation settings mismatch: {root}')
            results.append(evaluate_control_poses(row, profile, kind))
            analyze(profile)
            state['completed_tasks'].append({'condition': kind, 'id': row['id']})
            write_json(directory / 'comparison.json', results)
            plot_control_validation(results)
            write_json(directory / 'state.json', state)
        for row in controls:
            results.append(evaluate_control_poses(row, 'main', 'predicted_fullECD'))
        write_json(directory / 'comparison.json', results)
        plot_control_validation(results)
        state.update(status='completed', current_candidate=None, current_condition=None,
                     completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory / 'state.json', state)
    except BaseException as error:
        state.update(status='failed', error=str(error))
        write_json(directory / 'state.json', state)
        raise


def restore_pose_chains(source, destination, heavy_length):
    """Restore the biological H/L labels after docking; coordinates are unchanged."""
    lines = []
    source = Path(source)
    text = gzip.open(source, 'rt').read() if source.suffix == '.gz' else source.read_text()
    for line in text.splitlines():
        if line.startswith(('ATOM', 'HETATM')):
            chain = line[21]
            if chain == 'B':
                chain = 'H' if int(line[22:26]) <= heavy_length else 'L'
            if chain not in {'A', 'H', 'L'}:
                raise ValueError(f'Unexpected docked chain {chain}: {source}')
            line = line[:21] + chain + line[22:72].ljust(50) + chain.ljust(4) + line[76:]
            lines.append(line)
    Path(destination).write_text('\n'.join(lines) + '\nEND\n', encoding='ascii')


def dock(limit=None, ids=None, profile='pilot', sampling=64, refine=8, native_reference=False):
    manifest = read_json(OUTPUT / 'manifest.json')
    rows = manifest['candidates']
    if ids:
        rows = [next(r for r in rows if r['id'] == candidate) for candidate in ids]
    if limit:
        rows = rows[:limit]
    if refine > sampling:
        raise ValueError('refine must be <= sampling')
    for row in rows:
        root = prepare_docking(row, sampling, refine, profile, native_reference)
        if (root / 'completed.json').exists():
            print(f"Cached docking: {row['id']}", flush=True)
            continue
        if (root / 'run').exists():
            raise RuntimeError(f'Interrupted docking exists: {root}; inspect logs before HADDOCK --restart')
        print(f"Docking {profile}: {row['id']} {sampling}/{refine}", flush=True)
        run_logged([ENV / 'docking/bin/haddock3', 'workflow.cfg'], root / 'execution.log', cwd=root,
                   extra_env={'PATH': str(ENV / 'docking/bin') + ':' + os.environ.get('PATH', '')})
        write_json(root / 'completed.json', {'settings_sha256': digest(root / 'settings.json')})


def complete_rigidbody_pose(path, expected_atoms):
    """Accept only complete, finite CNS coordinates with all topology atoms and energy terms."""
    import math
    from haddock.gear.haddockmodel import HaddockModel
    lines = Path(path).read_text().splitlines()
    if not lines or lines[-1].strip() != 'END':
        raise ValueError('Missing PDB END record')
    atoms = [line for line in lines if line.startswith('ATOM')]
    if sorted(line[12:27] for line in atoms) != expected_atoms:
        raise ValueError('Incomplete or different topology atoms')
    if not all(math.isfinite(float(line[i:i+8])) for line in atoms for i in (30, 38, 46)):
        raise ValueError('Nonfinite coordinates')
    energies = HaddockModel(path).energies
    if not {'vdw', 'elec', 'desolv', 'air', 'bsa'} <= energies.keys():
        raise ValueError('Missing HADDOCK energy terms')
    if not all(math.isfinite(value) for value in energies.values()):
        raise ValueError('Nonfinite HADDOCK energies')
    return digest(path)


def prepare_rigidbody_resume(candidate, sampling, refine):
    """Back up verified complete poses before the official stage restart removes its folder."""
    import inspect
    from haddock.modules.sampling.rigidbody import HaddockModule
    row = next(r for r in read_json(OUTPUT / 'manifest.json')['candidates'] if r['id'] == candidate)
    root = prepare_docking(row, sampling, refine, 'main')
    if (root / 'completed.json').exists():
        return root
    stage = root / 'run/1_rigidbody'
    if not stage.is_dir() or not (root / 'run/0_topoaa/io.json').exists():
        raise RuntimeError('Only an interrupted rigidbody stage after completed topology can be resumed here')
    if any(p.is_dir() and re.match(r'[2-9]_', p.name) for p in (root / 'run').iterdir()):
        raise RuntimeError('Later docking stages exist; inspect their restart point first')
    checkpoint = read_json(OUTPUT / 'shutdown_checkpoint_latest.json')
    archive = OUTPUT / 'checkpoints' / Path(checkpoint['archive']).name
    if checkpoint['interrupted_candidate'] != candidate or checkpoint['interrupted_stage'] != '1_rigidbody' or digest(archive) != checkpoint['sha256']:
        raise RuntimeError('Shutdown archive does not match this interrupted candidate')
    expected_atoms = sorted(line[12:27] for path in (root / 'run/0_topoaa').glob('*.pdb')
                            for line in path.read_text().splitlines() if line.startswith('ATOM'))
    cache = root / 'resume_rigidbody_cache'
    cache.mkdir(exist_ok=True)
    reused, rejected = [], []
    for path in sorted(stage.glob('rigidbody_*.pdb')):
        index = int(path.stem.rsplit('_', 1)[1])
        if not 1 <= index <= sampling:
            raise ValueError(f'Unexpected pose number: {path}')
        try:
            sha = complete_rigidbody_pose(path, expected_atoms)
        except (ValueError, OSError) as error:
            rejected.append({'file': path.name, 'reason': str(error)})
            continue
        shutil.copy2(path, cache / path.name)
        if digest(cache / path.name) != sha:
            raise RuntimeError('Pose backup checksum mismatch')
        reused.append({'file': path.name, 'index': index, 'seed': 42 + index, 'sha256': sha})
    record = {'candidate': candidate, 'settings_sha256': digest(root / 'settings.json'),
              'source_archive_sha256': checkpoint['sha256'], 'reused': reused,
              'rejected': rejected, 'expected_total': sampling, 'remaining': sampling - len(reused),
              'factory_source_sha256': hashlib.sha256(inspect.getsource(HaddockModule.make_cns_jobs).encode()).hexdigest()}
    write_json(root / 'resume_rigidbody.json', record)
    print(f'Resume {candidate}: reuse {len(reused)} complete poses; compute {record["remaining"]}', flush=True)
    return root


def resume_rigidbody_worker(candidate, profile='main'):
    """Use official HADDOCK restart, retaining its original per-pose seeds and output metadata."""
    import inspect
    from haddock.clis.cli import main as haddock_main
    from haddock.modules.sampling.rigidbody import HaddockModule
    root = ENV / 'runs' / profile / candidate
    record = read_json(root / 'resume_rigidbody.json')
    original = HaddockModule.make_cns_jobs
    factory_sha = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    if factory_sha != record['factory_source_sha256'] or digest(root / 'settings.json') != record['settings_sha256']:
        raise RuntimeError('HADDOCK API or docking settings changed since resume validation')
    cached = {pose['index']: pose for pose in record['reused']}
    for pose in cached.values():
        if digest(root / 'resume_rigidbody_cache' / pose['file']) != pose['sha256']:
            raise RuntimeError('Cached pose checksum mismatch')

    def pending_jobs(module, inputs):
        jobs = original(module, inputs)
        if len(jobs) != record['expected_total']:
            raise RuntimeError('Unexpected number of official CNS jobs')
        pending = []
        for index, job in enumerate(jobs, 1):
            if index not in cached:
                pending.append(job)
                continue
            pose = cached[index]
            model = module.output_models[index - 1]
            if model.seed != pose['seed'] or str(model.file_name) != pose['file']:
                raise RuntimeError('Official CNS job identity/seed mismatch')
            shutil.copy2(root / 'resume_rigidbody_cache' / pose['file'], Path.cwd() / pose['file'])
        print(f'Official CNS jobs: {len(pending)} new; {len(cached)} verified outputs restored', flush=True)
        return pending

    # Patch only this worker process. The installed library and scientific protocol stay unchanged.
    HaddockModule.make_cns_jobs = pending_jobs
    try:
        os.chdir(root)
        haddock_main(root / 'workflow.cfg', restart=1)
    finally:
        HaddockModule.make_cns_jobs = original


def prepare_flexref_resume(candidate, sampling, refine):
    """Preserve complete partial refinement outputs with their original selected-model seeds."""
    import inspect
    import haddock.modules.refinement.flexref as flexref
    from haddock.libs.libontology import ModuleIO
    checkpoint = read_json(OUTPUT / 'shutdown_checkpoint_latest.json')
    archive = OUTPUT / 'checkpoints' / Path(checkpoint['archive']).name
    if checkpoint['interrupted_candidate'] != candidate or digest(archive) != checkpoint['sha256']:
        raise RuntimeError('Latest shutdown archive does not match this candidate')
    row = next(r for r in read_json(OUTPUT / 'manifest.json')['candidates'] if r['id'] == candidate)
    root = prepare_docking(row, sampling, refine, 'main')
    if (root / 'completed.json').exists():
        return root
    selected = root / 'run/2_seletop/io.json'
    io = ModuleIO(); io.load(selected)
    models = io.retrieve_models()
    if len(models) != refine or any(model.seed is None for model in models):
        raise RuntimeError('Missing original refinement inputs/seeds')
    expected_atoms = sorted(line[12:27] for p in (root / 'run/0_topoaa').glob('*.pdb')
                            for line in p.read_text().splitlines() if line.startswith('ATOM'))
    cache = root / 'resume_flexref_cache'; cache.mkdir(exist_ok=True)
    reused, rejected = [], []
    for p in sorted((root / 'run/3_flexref').glob('flexref_*.pdb')):
        index = int(p.stem.rsplit('_', 1)[1])
        if not 1 <= index <= refine:
            raise ValueError(f'Unexpected refinement index: {p}')
        model = models[index - 1]
        try:
            sha = complete_rigidbody_pose(p, expected_atoms)
            text = p.read_text()
            parent = re.search(r'^REMARK initial structure 1 - (.+)$', text, re.M)
            if not parent or Path(parent[1].strip()).name != str(model.file_name):
                raise ValueError('Refinement input model does not match selection order')
        except (ValueError, OSError) as error:
            rejected.append({'file': p.name, 'reason': str(error)})
            continue
        shutil.copy2(p, cache / p.name)
        if digest(cache / p.name) != sha:
            raise RuntimeError('Refinement backup checksum mismatch')
        reused.append({'index': index, 'file': p.name, 'sha256': sha,
                       'source_model': str(model.file_name), 'seed': model.seed})
    record = {'candidate': candidate, 'expected_total': refine, 'reused': reused, 'rejected': rejected,
              'remaining': refine - len(reused), 'settings_sha256': digest(root / 'settings.json'),
              'selection_io_sha256': digest(selected),
              'module_source_sha256': hashlib.sha256(inspect.getsource(flexref.HaddockModule._run).encode()).hexdigest()}
    write_json(root / 'resume_flexref.json', record)
    print(f'Resume flexref {candidate}: reuse {len(reused)}; compute {record["remaining"]}', flush=True)
    return root


def resume_flexref_worker(candidate):
    """Filter complete jobs only in this worker, leaving official output/scoring code intact."""
    import inspect
    import haddock.modules.refinement.flexref as flexref
    from haddock.clis.cli import main as haddock_main
    from haddock.libs.libcns import write_eval_line
    from haddock.libs.libontology import ModuleIO
    root = ENV / 'runs/main' / candidate
    record = read_json(root / 'resume_flexref.json')
    if digest(root / 'settings.json') != record['settings_sha256'] or digest(root / 'run/2_seletop/io.json') != record['selection_io_sha256']:
        raise RuntimeError('Refinement settings or original selection changed')
    if hashlib.sha256(inspect.getsource(flexref.HaddockModule._run).encode()).hexdigest() != record['module_source_sha256']:
        raise RuntimeError('Installed refinement module changed')
    io = ModuleIO(); io.load(root / 'run/2_seletop/io.json')
    models = io.retrieve_models()
    cached = {pose['index']: pose for pose in record['reused']}
    for pose in cached.values():
        if digest(root / 'resume_flexref_cache' / pose['file']) != pose['sha256']:
            raise RuntimeError('Cached refinement pose changed')
    original_engine = flexref.get_engine

    def filtered_engine(mode, params):
        if mode != 'local' or params['sampling_factor'] != 1:
            raise RuntimeError('Partial refinement resume requires local mode and sampling_factor 1')
        engine = original_engine(mode, params)
        def build(jobs):
            if len(jobs) != record['expected_total']:
                raise RuntimeError('Unexpected number of refinement jobs')
            pending = []
            for index, job in enumerate(jobs, 1):
                model = models[index - 1]
                script = job.input_file if isinstance(job.input_file, str) else Path(job.input_file).read_text()
                if Path(job.output_file).name != f'flexref_{index}.out' or str(model.file_name) not in script or write_eval_line('seed', model.seed) not in script:
                    raise RuntimeError('Official refinement job identity or inherited seed mismatch')
                if index in cached:
                    shutil.copy2(root / 'resume_flexref_cache' / cached[index]['file'], Path.cwd() / cached[index]['file'])
                else:
                    pending.append(job)
            print(f'Official flexref jobs: {len(pending)} new; {len(cached)} verified outputs restored', flush=True)
            return engine(pending)
        return build

    flexref.get_engine = filtered_engine
    try:
        os.chdir(root)
        haddock_main(root / 'workflow.cfg', restart=3)
    finally:
        flexref.get_engine = original_engine


def interface_metrics(pdb, residue_map):
    """Heavy-atom intermolecular contacts only; glycan-site proximity is a risk flag."""
    import numpy as np
    from scipy.spatial import cKDTree
    from Bio.PDB import PDBParser
    import freesasa
    model = PDBParser(QUIET=True).get_structure('pose', pdb)[0]
    if set(c.id for c in model) != {'A', 'H', 'L'}:
        raise ValueError(f'Unexpected docking chains: {pdb}')
    atoms = {group: [a for c in model if c.id in chains for a in c.get_atoms()
                     if a.element not in ('H', 'D')]
             for group, chains in [('antibody', {'H', 'L'}), ('antigen', {'A'})]}
    antibody, antigen = atoms['antibody'], atoms['antigen']
    tree = cKDTree([a.coord for a in antibody])
    neighbors = tree.query_ball_point([a.coord for a in antigen], r=4.0)
    pairs, ab_res, ag_res = set(), set(), set()
    cdrs = set((name[-1], i) for name, values in residue_map['cdr_positions'].items() for i in values)
    for ag_atom, contacts in zip(antigen, neighbors):
        for index in contacts:
            ab_atom = antibody[index]
            ab = (ab_atom.parent.parent.id, ab_atom.parent.id[1])
            ag = ag_atom.parent.id[1]
            pairs.add((ab[0], ab[1], ag))
            ab_res.add(ab)
            ag_res.add(ag)
    glycan_distances = {}
    for uniprot in (35, 192, 200, 219):
        if uniprot - 18 not in model['A']:
            glycan_distances[str(uniprot)] = None
            continue
        residue = model['A'][uniprot - 18]
        anchor = residue['ND2'] if 'ND2' in residue else residue['CA']
        glycan_distances[str(uniprot)] = float(tree.query(anchor.coord)[0])
    terminal = [a.coord for a in antigen if a.parent.id[1] >= 211]
    terminal_distance = float(np.min(tree.query(terminal)[0])) if terminal else None
    structures = {name: freesasa.Structure() for name in ('antibody', 'antigen', 'complex')}
    for group, group_atoms in atoms.items():
        for atom in group_atoms:
            residue = atom.parent
            for name in (group, 'complex'):
                structures[name].addAtom(atom.fullname, residue.resname, str(residue.id[1]),
                                         residue.parent.id, *[float(x) for x in atom.coord])
    sasa = {name: float(freesasa.calc(structure).totalArea()) for name, structure in structures.items()}
    buried_total = sasa['antibody'] + sasa['antigen'] - sasa['complex']
    return {'heavy_atom_contact_count': sum(map(len, neighbors)), 'residue_contact_count': len(pairs),
            'antigen_resolved_residues_uniprot': sorted(r.id[1] + 18 for r in model['A'] if r.id[0] == ' '),
            'buried_SASA_total_A2': buried_total, 'interface_area_half_buried_SASA_A2': buried_total / 2,
            'contact_pairs': sorted([list(x) for x in pairs]),
            'antigen_contact_residues_uniprot': sorted(i + 18 for i in ag_res),
            'antibody_contact_residue_count': len(ab_res),
            'cdr_contact_residue_fraction': len(ab_res & cdrs) / len(ab_res) if ab_res else None,
            'glycan_anchor_distance_A': glycan_distances,
            'glycan_proximity_flag_8A': any(d is not None and d < 8 for d in glycan_distances.values()),
            'ECD_terminal10_distance_A': terminal_distance,
            'membrane_proximity_surrogate_flag_8A': terminal_distance < 8 if terminal_distance is not None else None}


def reference_controls():
    """Measure the resolved experimental Fv interfaces separately from de novo docking."""
    import numpy as np
    from scipy.spatial import cKDTree
    from Bio import Align
    from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain
    from Bio.SeqUtils import seq1
    manifest = read_json(OUTPUT / 'manifest.json')
    aligner = Align.PairwiseAligner(mode='global', match_score=2, mismatch_score=-2,
                                    open_gap_score=-10, extend_gap_score=-.5)

    def aligned_chain(chain, target, name, offset=0):
        residues = [r for r in chain if r.id[0] == ' ' and 'CA' in r]
        sequence = ''.join(seq1(r.resname) for r in residues)
        indices = aligner.align(target, sequence)[0].indices
        mapping = [(int(i), residues[int(j)]) for i, j in indices.T if i >= 0 and j >= 0]
        matches = sum(seq1(r.resname) == target[i] for i, r in mapping)
        if matches < min(90, int(.85 * len(target))):
            raise ValueError(f'Cannot align reference chain {chain.id} to {name}')
        new = Chain.Chain(name)
        mismatch = []
        for i, residue in mapping:
            copy_residue = residue.copy()
            copy_residue.id = (' ', i + 1 + offset, ' ')
            copy_residue.segid = name
            new.add(copy_residue)
            if seq1(residue.resname) != target[i]:
                mismatch.append({'sequence_index': i + 1, 'expected': target[i], 'observed': seq1(residue.resname)})
        return new, mismatch

    results = []
    for row in manifest['candidates']:
        if row['group'] != 'control':
            continue
        source = OUTPUT / 'references' / f"{row['reference_pdb']}.pdb"
        model = PDBParser(QUIET=True).get_structure('reference', source)[0]
        heavy_id = row['reference_domains']['H']['source_chains'][0]
        heavy, heavy_mismatch = aligned_chain(model[heavy_id], row['VH_AA'], 'H')
        heavy_tree = cKDTree([a.coord for a in heavy.get_atoms() if a.name == 'CA'])
        light_candidates = []
        for chain_id in row['reference_domains']['L']['source_chains']:
            light, mismatch = aligned_chain(model[chain_id], row['VL_AA'], 'L', len(row['VH_AA']))
            distances = heavy_tree.query([a.coord for a in light.get_atoms() if a.name == 'CA'])[0]
            light_candidates.append((int(np.sum(distances < 10)), chain_id, light, mismatch))
        _, light_id, light, light_mismatch = max(light_candidates, key=lambda x: x[0])
        fv_tree = cKDTree([a.coord for c in [heavy, light] for a in c.get_atoms() if a.element != 'H'])
        antibody_ids = set(chain_id for d in row['reference_domains'].values() for chain_id in d['source_chains'])
        antigen_candidates = []
        for chain in model:
            if chain.id in antibody_ids:
                continue
            try:
                antigen, mismatch = aligned_chain(chain, manifest['antigen'], 'A')
            except ValueError:
                continue
            distances = fv_tree.query([a.coord for a in antigen.get_atoms() if a.element != 'H'])[0]
            antigen_candidates.append((int(np.sum(distances < 4)), chain.id, antigen, mismatch))
        if not antigen_candidates:
            raise ValueError(f'Cannot find PD-L1 chain in {source}')
        _, antigen_id, antigen, antigen_mismatch = max(antigen_candidates, key=lambda x: x[0])
        structure = Structure.Structure('cropped_reference')
        structure.add(Model.Model(0))
        for chain in [heavy, light, antigen]:
            structure[0].add(chain)
        destination = OUTPUT / 'references' / row['id']
        destination.mkdir(exist_ok=True)
        pdb = destination / 'resolved_Fv_PDL1.pdb'
        writer = PDBIO()
        writer.set_structure(structure)
        writer.save(str(pdb))
        cdrs = cdr_positions(row)
        cdrs = {name: [i + (len(row['VH_AA']) if name.endswith('.L') else 0) for i in values]
                for name, values in cdrs.items()}
        metrics = interface_metrics(pdb, {'cdr_positions': cdrs})
        log = destination / 'prodigy.log'
        run_logged([ENV / 'docking/bin/prodigy', pdb, '--selection', 'H,L', 'A',
                    '--temperature', '25', '--distance-cutoff', '5.5'], log)
        result = {'id': row['id'], 'reference_pdb': row['reference_pdb'],
                  'source_chains_H_L_antigen': [heavy_id, light_id, antigen_id],
                  'resolved_chain_lengths': {c.id: len(c) for c in structure[0]},
                  'sequence_mismatches': {'H': heavy_mismatch, 'L': light_mismatch, 'A': antigen_mismatch},
                  'interface': metrics, 'prodigy_raw_log': str(log),
                  'scope': 'Resolved experimental crystal interface; NOT ColabFold/docking output and NOT a measured Kd.'}
        write_json(destination / 'reference_metrics.json', result)
        results.append(result)
    write_json(OUTPUT / 'references' / 'reference_controls_summary.json', results)
    print('Experimental reference interfaces analyzed separately: 2', flush=True)


def analyze(profile):
    import numpy as np
    import pandas as pd
    from collections import Counter
    manifest = read_json(OUTPUT / 'manifest.json')
    summary, excluded = [], []
    destination = OUTPUT / 'analysis' / profile
    destination.mkdir(parents=True, exist_ok=True)
    expected_rows = [r for r in manifest['candidates'] if not profile.startswith('reference-pilot') or r['group'] == 'control']
    for row in expected_rows:
        root = ENV / 'runs' / profile / row['id']
        if not (root / 'completed.json').exists():
            excluded.append({'id': row['id'], 'reason': 'docking not completed'})
            continue
        tables = sorted((root / 'run').glob('*_caprieval/capri_ss.tsv'))
        if not tables:
            excluded.append({'id': row['id'], 'reason': 'missing caprieval output'})
            continue
        table_path = tables[-1]
        table = pd.read_csv(table_path, sep='\\t', engine='python', comment='#')
        if 'cluster_ranking' not in table:
            raise ValueError(f'Unexpected output columns: {table_path}: {list(table)}')
        top = table[pd.to_numeric(table['cluster_ranking'], errors='coerce') == 1].sort_values('score').head(4)
        if top.empty:
            excluded.append({'id': row['id'], 'reason': 'no cluster available; no silent fallback'})
            continue
        metrics = []
        maps = read_json(root / 'residue_map.json')
        candidate_dir = destination / row['id']
        candidate_dir.mkdir(exist_ok=True)
        cluster_tables = list((root / 'run').glob('*_clustfcc/clustfcc.tsv'))
        cluster_population = None
        if cluster_tables:
            clustered = pd.read_csv(cluster_tables[-1], sep='\t', comment='#')
            cluster_population = int((pd.to_numeric(clustered['cluster_id'], errors='coerce') ==
                                       float(top.iloc[0]['cluster_id'])).sum())
        if profile == 'main' and (cluster_population is None or cluster_population < 4):
            excluded.append({'id': row['id'], 'reason': 'cluster has fewer than four refined poses'})
            continue
        analysis_signature = {'version': 3, 'settings_sha256': digest(root / 'settings.json'),
                              'capri_sha256': digest(table_path), 'residue_map_sha256': digest(root / 'residue_map.json')}
        cached_metrics = candidate_dir / 'metrics.json'
        if cached_metrics.exists() and read_json(cached_metrics).get('analysis_signature') == analysis_signature:
            summary.append(read_json(cached_metrics))
            continue
        for rank, (_, pose) in enumerate(top.iterrows(), 1):
            pdb = (table_path.parent / str(pose['model'])).resolve()
            if not pdb.exists() and pdb.with_suffix(pdb.suffix + '.gz').exists():
                pdb = pdb.with_suffix(pdb.suffix + '.gz')
            if not pdb.exists():
                raise FileNotFoundError(pdb)
            local = candidate_dir / f'pose_{rank}.pdb'
            restore_pose_chains(pdb, local, len(row['VH_AA']))
            log = candidate_dir / f'prodigy_{rank}.log'
            signature = {'pdb_sha256': digest(local), 'selection': ['H,L', 'A'],
                         'temperature_C': 25, 'cutoff_A': 5.5}
            cache = log.with_suffix('.result.json')
            if cache.exists():
                prediction = read_json(cache)
                if prediction['signature'] != signature:
                    raise RuntimeError(f'PRODIGY cache mismatch: {cache}')
            else:
                run_logged([ENV / 'docking/bin/prodigy', local, '--selection', 'H,L', 'A',
                            '--temperature', '25', '--distance-cutoff', '5.5'], log)
                text = log.read_text()
                affinity = re.search(r'Predicted binding affinity.*?:\s*([-+\d.eE]+)', text)
                kd = re.search(r'Predicted dissociation constant.*?:\s*([-+\d.eE]+)', text)
                if not affinity or not kd:
                    raise ValueError(f'Cannot parse PRODIGY output: {log}')
                prediction = {'signature': signature, 'deltaG_kcal_mol': float(affinity[1]),
                              'estimated_Kd_M': float(kd[1])}
                write_json(cache, prediction)
            interface = interface_metrics(local, maps)
            if row['group'] == 'control':
                reference = read_json(OUTPUT / 'references' / row['id'] / 'reference_metrics.json')['interface']
                native_pairs = set(tuple(x) for x in reference['contact_pairs'])
                predicted_pairs = set(tuple(x) for x in interface['contact_pairs'])
                interface['reference_contact_recall_4A'] = len(native_pairs & predicted_pairs) / len(native_pairs) if native_pairs else None
                interface['reference_contact_precision_4A'] = len(native_pairs & predicted_pairs) / len(predicted_pairs) if predicted_pairs else None
                interface['reference_contact_Jaccard_4A'] = len(native_pairs & predicted_pairs) / len(native_pairs | predicted_pairs) if native_pairs | predicted_pairs else None
            metrics.append({'pose': rank, 'score': float(pose['score']),
                            'prodigy_deltaG_kcal_mol': prediction['deltaG_kcal_mol'],
                            'prodigy_estimated_Kd_M': prediction['estimated_Kd_M'],
                            'interface': interface})
        frequencies = Counter(i for m in metrics for i in m['interface']['antigen_contact_residues_uniprot'])
        result = {'id': row['id'], 'group': row['group'], 'profile': profile,
                  'analysis_signature': analysis_signature,
                  'cluster_id': str(top.iloc[0]['cluster_id']),
                  'cluster_top4_n': len(metrics),
                  'cluster_population': cluster_population,
                  'underpopulated_cluster': cluster_population is None or cluster_population < 4,
                  'refined_pose_count': read_json(root / 'settings.json')['refine'],
                  'cluster_top4_score_mean': float(top['score'].mean()),
                  'cluster_top4_score_sd': float(np.std(top['score'])),
                  'prodigy_top4_deltaG_mean': float(np.mean([m['prodigy_deltaG_kcal_mol'] for m in metrics])),
                  'contact_frequency_top4': {str(i): n / len(metrics) for i, n in sorted(frequencies.items())},
                  'representative': 'pose_1.pdb', 'poses': metrics,
                  'pilot_only': profile != 'main',
                  'native_reference': read_json(root / 'settings.json')['native_reference']}
        write_json(candidate_dir / 'metrics.json', result)
        summary.append(result)
        # A directly executable PyMOL scene for the representative pose.
        (candidate_dir / 'view.pml').write_text('load pose_1.pdb, complex\nhide everything\n'
            'show cartoon, complex\ncolor gray80, chain A\ncolor marine, chain H\ncolor cyan, chain L\n'
            'select interface, byres ((chain H+L within 4 of chain A) or (chain A within 4 of chain H+L))\n'
            'show sticks, interface\nset cartoon_transparency, 0.1\nbg_color white\nzoom complex\n', encoding='ascii')
    write_json(destination / 'all_results.json', {'results': summary, 'excluded': excluded,
               'expected_candidates': len(expected_rows), 'interpretation': 'Computational pose-based estimates; no experimental affinity claim.'})
    fields = ['id', 'group', 'cluster_top4_score_mean', 'cluster_top4_score_sd', 'prodigy_top4_deltaG_mean', 'pilot_only']
    with (destination / 'candidate_summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in summary)
    if summary:
        plot_results(summary, destination, profile, excluded)
        shortlist(summary, destination)
        plot_poses(summary, destination, profile)
    print(f'Analyzed {len(summary)}/{len(expected_rows)}; pending/unclustered {len(excluded)}; profile={profile}', flush=True)


def plot_results(rows, destination, profile, excluded):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors = {'MAGE': '#64748b', 'LoRA': '#0f8575', 'control': '#db8c24'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    rng = np.random.default_rng(42)
    for axis, key, title, label in [(axes[0], 'cluster_top4_score_mean', 'Top-cluster docking score', 'HADDOCK score (a.u.; lower is better)'),
                                   (axes[1], 'prodigy_top4_deltaG_mean', 'Auxiliary PRODIGY estimate', 'Predicted delta G (kcal/mol)')]:
        for index, group in enumerate(colors):
            values = [r[key] for r in rows if r['group'] == group]
            if values:
                axis.scatter(index + rng.uniform(-0.12, 0.12, len(values)), values,
                             color=colors[group], edgecolor='white', s=38, zorder=3)
                axis.plot([index - .2, index + .2], [np.median(values)] * 2, color='black', lw=2)
        axis.set(xticks=range(3), xticklabels=list(colors), title=title, ylabel=label)
        axis.set_xlim(-.5, 2.5)
        axis.grid(axis='y', alpha=.18)
    for group, color in colors.items():
        group_rows = [r for r in rows if r['group'] == group]
        if group_rows:
            positions = np.arange(19, 239)
            values = []
            for i in positions:
                observed = [r['contact_frequency_top4'].get(str(i), 0) for r in group_rows
                            if i in r['poses'][0]['interface']['antigen_resolved_residues_uniprot']]
                values.append(np.mean(observed) if observed else np.nan)
            axes[2].plot(positions, values, label=f'{group} (n={len(group_rows)})', color=color)
    axes[2].set(xlabel='PD-L1 residue (UniProt numbering)', ylabel='Mean top-cluster contact frequency', title='Antigen interface contacts (4 A)')
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(f'PD-L1 structural screening | {profile} | {len(rows)}/{len(rows)+len(excluded)} analyzed')
    fig.supxlabel(f'One point per antibody; up to four cluster poses. Missing/unclustered: {len(excluded)}. Predictions do not establish binding.', fontsize=9)
    for suffix in ('png', 'svg'):
        fig.savefig(destination / f'docking_summary.{suffix}', dpi=200)
    plt.close(fig)


def shortlist(rows, destination):
    """Descriptive ranking; flags remain visible and no experimental affinity is inferred."""
    ranked = []
    for group in ('MAGE', 'LoRA', 'control'):
        selected = sorted([r for r in rows if r['group'] == group], key=lambda r:r['cluster_top4_score_mean'])
        for rank, row in enumerate(selected, 1):
            poses = row['poses']
            ranked.append({'id': row['id'], 'group': group, 'rank_within_group': rank,
                           'HADDOCK_cluster_score_mean': row['cluster_top4_score_mean'],
                           'HADDOCK_cluster_score_sd': row['cluster_top4_score_sd'],
                           'cluster_population': row['cluster_population'],
                           'scored_poses': row['cluster_top4_n'],
                           'PRODIGY_deltaG_mean_kcal_mol': row['prodigy_top4_deltaG_mean'],
                           'interface_area_A2_mean': sum(p['interface']['interface_area_half_buried_SASA_A2'] for p in poses)/len(poses),
                           'CDR_contact_fraction_mean': sum(p['interface']['cdr_contact_residue_fraction'] or 0 for p in poses)/len(poses),
                           'glycan_proximity_flag': any(p['interface']['glycan_proximity_flag_8A'] for p in poses),
                           'ECD_terminal_proximity_flag': any(p['interface']['membrane_proximity_surrogate_flag_8A'] for p in poses),
                           'underpopulated_cluster': row['underpopulated_cluster'],
                           'pilot_only': row['pilot_only'],
                           'representative': f"{row['id']}/pose_1.pdb"})
    for filename, values in [('ranking_all.csv', ranked),
                             ('shortlist_top5_per_group.csv', [r for r in ranked if r['rank_within_group'] <= 5])]:
        with (destination / filename).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
            writer.writeheader()
            writer.writerows(values)


def plot_poses(rows, destination, profile):
    """Export comparable C-alpha traces; colors show chains, not model confidence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    from Bio.PDB import PDBParser
    from Bio.SVDSuperimposer import SVDSuperimposer
    selected = []
    for group in ('MAGE', 'LoRA', 'control'):
        selected.extend(sorted([r for r in rows if r['group'] == group],
                               key=lambda r:r['cluster_top4_score_mean'])[:2])
    if not selected:
        return
    colors = {'A': '#a4adb9', 'H': '#2263b2', 'L': '#0f9c96'}
    parsed = []
    reference = None
    for row in selected:
        model = PDBParser(QUIET=True).get_structure(row['id'], destination / row['id'] / 'pose_1.pdb')[0]
        ca = {r.id[1]: r['CA'].coord for r in model['A'] if 'CA' in r}
        if reference is None:
            reference = ca
        common = sorted(set(reference) & set(ca))
        if len(common) < 3:
            raise ValueError('Insufficient common antigen residues for structure-panel alignment')
        fit = SVDSuperimposer()
        fit.set(np.array([reference[i] for i in common]), np.array([ca[i] for i in common]))
        fit.run()
        rotation, translation = fit.get_rotran()
        traces = {chain: np.array([r['CA'].coord for r in model[chain] if 'CA' in r]) @ rotation + translation
                  for chain in colors}
        parsed.append((row, traces))
    points = np.concatenate([trace for _, traces in parsed for trace in traces.values()])
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    radius = float(np.max(points.max(axis=0) - points.min(axis=0))) / 2 * 1.05
    cols = min(3, len(parsed))
    nrows = (len(parsed) + cols - 1) // cols
    fig = plt.figure(figsize=(max(8.4, 4.2 * cols), 4.6 * nrows + 1.0))
    for index, (row, traces) in enumerate(parsed, 1):
        axis = fig.add_subplot(nrows, cols, index, projection='3d')
        for chain, coords in traces.items():
            splits = np.where(np.linalg.norm(np.diff(coords, axis=0), axis=1) > 5)[0] + 1
            for segment in np.split(coords, splits):
                axis.plot(*segment.T, color=colors[chain], lw=1.8)
        axis.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
                 zlim=(center[2]-radius, center[2]+radius))
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=18, azim=35)
        axis.set_axis_off()
        axis.set_title(f"{row['id']}\nCluster score {row['cluster_top4_score_mean']:.1f}", fontsize=10, pad=10)
    fig.legend([Line2D([0],[0], color=color, lw=3) for color in colors.values()],
               ['PD-L1', 'VH', 'VL'], loc='lower center', ncol=3, frameon=False)
    fig.suptitle(f'PD-L1 representative docking poses | {profile}\nC-alpha traces; aligned on common antigen residues', fontsize=11, y=.97)
    fig.subplots_adjust(top=.78 if nrows == 1 else .87, bottom=.12, hspace=.15)
    for extension in ('png', 'svg'):
        fig.savefig(destination / f'representative_poses.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def import_colab(archive_path):
    """Import a Colab result archive only after checking the fixed biological inputs/settings."""
    import zipfile
    archive_path = Path(archive_path).resolve()
    expected = read_json(OUTPUT / 'manifest.json')
    with zipfile.ZipFile(archive_path) as archive:
        incoming = json.loads(archive.read('manifest.json'))
        for key in ('input_hashes', 'antigen', 'candidates'):
            if incoming[key] != expected[key]:
                raise ValueError(f'Colab input mismatch: {key}')
        if incoming['settings']['fold'] != expected['settings']['fold']:
            raise ValueError('Colab folding settings do not match')
        for entry in archive.infolist():
            relative = Path(entry.filename)
            target = (OUTPUT / relative).resolve()
            if not target.is_relative_to(OUTPUT.resolve()) or relative.is_absolute():
                raise ValueError(f'Unsafe archive path: {entry.filename}')
            if relative.parts and relative.parts[0] in {'references', 'analysis', 'docking'}:
                raise ValueError('Folding archive cannot overwrite local docking/analysis')
        local_target = OUTPUT / 'folding/PDL1_ECD'
        backup = OUTPUT / 'local_before_colab/PDL1_ECD'
        if local_target.exists() and not backup.exists():
            shutil.copytree(local_target, backup)
        archive.extractall(OUTPUT)
    write_json(OUTPUT / 'colab_import.json', {'archive': str(archive_path), 'sha256': digest(archive_path),
               'imported_utc': datetime.now(timezone.utc).isoformat()})
    status()
    folding_summary()


def run_main(sampling, refine):
    """Run the common docking budget, reusing exact completed candidates, with balanced order."""
    manifest = read_json(OUTPUT / 'manifest.json')
    controls = [r['id'] for r in manifest['candidates'] if r['group'] == 'control']
    mage = [r['id'] for r in manifest['candidates'] if r['group'] == 'MAGE']
    lora = [r['id'] for r in manifest['candidates'] if r['group'] == 'LoRA']
    order = controls + [x for pair in zip(mage, lora) for x in pair]
    missing = [x for x in ['PDL1_ECD'] + order if not (OUTPUT / 'folding' / x / 'completed.json').exists()]
    if missing:
        raise RuntimeError(f'Collect completed Colab structures before docking: {missing}')
    settings = {'seed': 42, 'sampling': sampling, 'refine': refine, 'order': order,
                'epitope': 'all accessible PD-L1 ECD surface', 'status': 'running'}
    saved = OUTPUT / 'main_settings.json'
    if saved.exists():
        previous = read_json(saved)
        if any(previous[k] != settings[k] for k in ('seed', 'sampling', 'refine', 'order')):
            raise ValueError('Cannot mix main docking sampling budgets')
    write_json(saved, settings)
    for candidate in order:
        dock(ids=[candidate], profile='main', sampling=sampling, refine=refine)
        analyze('main')
        status()
    settings['status'] = 'completed'
    write_json(saved, settings)


def status():
    manifest = read_json(OUTPUT / 'manifest.json')
    names = ['PDL1_ECD'] + [r['id'] for r in manifest['candidates']]
    completed = [name for name in names if (OUTPUT / 'folding' / name / 'completed.json').exists()]
    result = {'folding_completed': len(completed), 'folding_total': len(names),
              'completed_ids': completed, 'pending_ids': [x for x in names if x not in completed],
              'docking_completed': [str(p.parent) for p in (ENV / 'runs').glob('*/*/completed.json')]}
    write_json(OUTPUT / 'pipeline_status.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def folding_summary():
    import numpy as np
    manifest = read_json(OUTPUT / 'manifest.json')
    results = []
    for row in manifest['candidates']:
        directory = OUTPUT / 'folding' / row['id']
        if not (directory / 'completed.json').exists():
            continue
        scores = list(directory.glob('*_scores_rank_001*.json'))
        if len(scores) != 1:
            raise ValueError(f'Unexpected rank-1 confidence files: {directory}')
        confidence = read_json(scores[0])
        plddt = np.asarray(confidence['plddt'])
        heavy = len(row['VH_AA'])
        if len(plddt) != heavy + len(row['VL_AA']):
            raise ValueError(f'Confidence length mismatch: {row["id"]}')
        cdrs = cdr_positions(row)
        result = {'id': row['id'], 'group': row['group'], 'VH_mean_pLDDT': float(np.mean(plddt[:heavy])),
                  'VL_mean_pLDDT': float(np.mean(plddt[heavy:])),
                  'CDR_mean_pLDDT': {name: float(np.mean(plddt[np.array(indices) - 1 +
                                      (heavy if name.endswith('.L') else 0)])) for name, indices in cdrs.items()},
                  'pdb_sha256': digest(predicted_structure(row['id']))}
        if 'pae' in confidence:
            pae = np.asarray(confidence['pae'])
            result['H_to_L_mean_PAE_A'] = float(np.mean(pae[:heavy, heavy:]))
            result['L_to_H_mean_PAE_A'] = float(np.mean(pae[heavy:, :heavy]))
        results.append(result)
    write_json(OUTPUT / 'folding_summary.json', {'completed_Fv': len(results), 'expected_Fv': 42,
               'interpretation': 'Fv fold confidence only; these metrics do not evaluate PD-L1 binding.', 'results': results})
    print(f'Fv confidence: {len(results)}/42', flush=True)


def make_colab():
    """Export a small notebook and a hash-verified bundle of exact code/inputs/cached MSA."""
    import io
    import zipfile
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(__file__, '구조평가.py')
        archive.write(OUTPUT / 'manifest.json', 'results/manifest.json')
        for path in (OUTPUT / 'inputs').glob('*.fasta'):
            archive.write(path, 'results/inputs/' + path.name)
        for path in (OUTPUT / 'folding').rglob('*.a3m'):
            archive.write(path, 'results/' + str(path.relative_to(OUTPUT)))
    payload_sha = hashlib.sha256(payload.getvalue()).hexdigest()
    bundle = HERE / '구조평가_환경/colab_inputs.zip'
    bundle.write_bytes(payload.getvalue())
    def cell(kind, source):
        result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            result.update(execution_count=None, outputs=[])
        return result
    setup = '''import os, sys, subprocess, pathlib, json, time
from pathlib import Path
os.environ['MPLBACKEND'] = 'Agg'
gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(gpu.stdout)
assert gpu.returncode == 0, '런타임 유형을 GPU로 변경한 후 실행하세요. CPU로 계산하지 않습니다.'
ENV = Path('/content/pdl1-env')
python = ENV/'colabfold/bin/python'
def checked(args):
    result = subprocess.run([str(x) for x in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout[-8000:])
    result.check_returncode()
if not python.exists():
    checked([sys.executable, '-m', 'pip', 'install', '-q', 'virtualenv'])
    checked([sys.executable, '-m', 'virtualenv', str(ENV/'colabfold')])
checked([python, '-m', 'pip', 'install', 'colabfold[alphafold,openmm]==1.6.3', 'jax[cuda12]==0.10.1'])
checked([python, '-c', 'import jax; print(jax.devices()); assert jax.default_backend()=="gpu"'])
'''
    unpack = f'''import io, zipfile, hashlib, urllib.request, urllib.parse, shutil
url = 'https://raw.githubusercontent.com/sanghyeb2/PDL1_Antibodygenerator_project/MAGE_ROLA/' + urllib.parse.quote('항체실험/구조평가_환경/colab_inputs.zip')
payload = urllib.request.urlopen(url).read()
assert hashlib.sha256(payload).hexdigest() == {payload_sha!r}
ROOT = Path('/content/pdl1-project')
ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(ROOT)
OUTPUT = ROOT/'results'
os.environ['PDL1_STRUCTURE_OUTPUT'] = str(OUTPUT)
os.environ['PDL1_STRUCTURE_ENV'] = str(ENV)
shutil.copy2(ROOT/'구조평가.py', OUTPUT/'execution_runner.py')
(OUTPUT/'colab_gpu.txt').write_text(gpu.stdout)
with (OUTPUT/'colab_requirements_lock.txt').open('w') as handle:
    subprocess.run([str(python), '-m', 'pip', 'freeze'], stdout=handle, check=True)
print('고정 입력: Fv 42쌍 + 공통 PD-L1, 시드 42; 정확한 코드 SHA256:', hashlib.sha256((ROOT/'구조평가.py').read_bytes()).hexdigest())
'''
    predict = '''# 완료된 구조는 재사용하고 실제 예측의 최신 로그를 표시합니다.
import asyncio
os.environ['MPLBACKEND'] = 'Agg'
async def run_stage(mode, *args):
    command = [str(python), str(ROOT/'구조평가.py'), mode, *args]
    log = OUTPUT / (mode + '_driver.log')
    previous = None
    with log.open('w') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        (OUTPUT/(mode + '_process.json')).write_text(json.dumps({'pid': process.pid, 'command': command, 'detached_session': True}))
        print('시작:', mode, 'PID:', process.pid, flush=True)
        while process.poll() is None:
            logs = list((OUTPUT/'folding').glob('*/log.txt'))
            if logs:
                latest = max(logs, key=lambda p:p.stat().st_mtime)
                lines = latest.read_text(errors='replace').splitlines()
                message = lines[-1] if lines else '모델 준비 중'
                if message != previous:
                    print(latest.parent.name, message, flush=True)
                    previous = message
            await asyncio.sleep(20)
        if process.returncode:
            print(log.read_text()[-6000:])
            raise RuntimeError('Stage failed: ' + mode)
    print('완료:', mode, flush=True)
await run_stage('fold', '--ids', 'PDL1_ECD')
# Save the completed common target before starting the longer Fv batch.
from google.colab import files
target_archive = shutil.make_archive('/content/PDL1_target_checkpoint_seed42', 'zip', OUTPUT)
files.download(target_archive)
await run_stage('fold-batch')
await run_stage('fold-summary')
'''
    export = '''# 모든 결과/로그를 로컬로 내려받아 HADDOCK3 단계에서 재사용합니다.
import shutil
from google.colab import files
archive = shutil.make_archive('/content/PDL1_ColabFold_results_seed42', 'zip', OUTPUT)
print('결과:', archive)
files.download(archive)
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'colab': {'name': 'PDL1_ColabFold_42Fv_seed42.ipynb'}, 'accelerator': 'GPU'},
                'cells': [cell('markdown', '# PD-L1: MAGE 20 + LoRA 20 + 양성 대조군 2쌍\n'
                               '런타임 → 런타임 유형 변경 → GPU를 선택합니다. 입력·코드는 우리 GitHub의 SHA256 검증 묶음으로 고정합니다.\n'
                               'Colab/공식 MMseqs2로 서열 전송은 사용자가 2026-11-14 승인했습니다. 유료 상품 가입·구매는 하지 않습니다.\n'
                               '출력은 결합 예측이 아닌 Fv/표적 구조이며, 이후 HADDOCK3 및 PRODIGY 단계가 필요합니다.\n'),
                          cell('code', setup), cell('code', unpack), cell('code', predict), cell('code', export)]}
    path = HERE / '구조평가_Colab.ipynb'
    write_json(path, notebook)
    print(f'Created {path} ({path.stat().st_size:,} bytes), payload SHA256 {payload_sha}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'fold', 'fold-batch', 'dock', 'analyze', 'status', 'fold-summary', 'reference-controls', 'validate-controls', 'audit-controls', 'prepare-controlled-controls', 'make-colab', 'import-colab', 'run-main', 'resume-main', 'resume-rigidbody-worker', 'resume-flexref', 'resume-flexref-worker'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--profile', default='pilot')
    parser.add_argument('--sampling', type=int, default=64)
    parser.add_argument('--refine', type=int, default=8)
    parser.add_argument('--native-reference', action='store_true')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'fold':
        fold(args.limit, args.ids)
    elif args.mode == 'fold-batch':
        fold_batch(args.ids)
    elif args.mode == 'dock':
        dock(args.limit, args.ids, args.profile, args.sampling, args.refine, args.native_reference)
    elif args.mode == 'analyze':
        analyze(args.profile)
    elif args.mode == 'fold-summary':
        folding_summary()
    elif args.mode == 'reference-controls':
        reference_controls()
    elif args.mode == 'validate-controls':
        validate_controls()
    elif args.mode == 'audit-controls':
        audit_controls()
    elif args.mode == 'prepare-controlled-controls':
        prepare_controlled_controls()
    elif args.mode == 'make-colab':
        make_colab()
    elif args.mode == 'import-colab':
        if args.archive is None:
            parser.error('--archive is required for import-colab')
        import_colab(args.archive)
    elif args.mode in {'resume-rigidbody-worker', 'resume-flexref-worker'}:
        if not args.ids or len(args.ids) != 1:
            parser.error('Exactly one --ids candidate is required')
        (resume_flexref_worker if args.mode == 'resume-flexref-worker' else resume_rigidbody_worker)(args.ids[0])
    elif args.mode in {'run-main', 'resume-main', 'resume-flexref'}:
        if args.sampling < 1000 or args.refine < 100:
            parser.error('Main docking requires explicit --sampling >=1000 --refine >=100; pilot defaults cannot be used.')
        if args.mode in {'resume-main', 'resume-flexref'}:
            if not args.ids or len(args.ids) != 1:
                parser.error('Exactly one interrupted --ids candidate is required')
            is_flexref = args.mode == 'resume-flexref'
            root = (prepare_flexref_resume if is_flexref else prepare_rigidbody_resume)(args.ids[0], args.sampling, args.refine)
            main_state = read_json(OUTPUT / 'main_settings.json')
            main_state['status'] = 'running'
            main_state.pop('processes_terminated', None)
            write_json(OUTPUT / 'main_settings.json', main_state)
            if not (root / 'completed.json').exists():
                run_logged([sys.executable, Path(__file__).resolve(), 'resume-flexref-worker' if is_flexref else 'resume-rigidbody-worker', '--ids', args.ids[0]],
                           root / ('resume_flexref_execution.log' if is_flexref else 'resume_execution.log'),
                           extra_env={'PATH': str(ENV / 'docking/bin') + ':' + os.environ.get('PATH', '')})
                write_json(root / 'completed.json', {'settings_sha256': digest(root / 'settings.json'),
                           'resumed': True, 'resume_metadata_sha256': digest(root / ('resume_flexref.json' if is_flexref else 'resume_rigidbody.json'))})
        run_main(args.sampling, args.refine)
    else:
        status()


if __name__ == '__main__':
    main()
