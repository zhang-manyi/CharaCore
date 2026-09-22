"""Offline source audit, not a split exporter, judge, or semantic classifier.

Curated decisions are read from docs; quotations and complete ledgers stay in runs.
No network access, candidate generation, policy profiles, or formal freeze output.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
REVISION = '290bf4ad22076156083804013012847a77c0646c'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def verify_source(root, row):
    path = (root / row['cache_path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Source outside project')
    data = path.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if sha(data) != row['sha256'] or blob != row['git_blob'] or len(data) != row['bytes']:
        raise ValueError('Source integrity mismatch: ' + row['path'])
    return data.decode('utf-8').splitlines()


def resolve_citation(citation, sources, texts):
    path = citation['source_path']
    if path not in sources or citation['source_sha256'] != sources[path]['sha256']:
        raise ValueError('Citation source mismatch: ' + path)
    lines = citation['lines']
    if not lines or lines != sorted(set(lines)) or any(
        type(n) is not int or not 1 <= n <= len(texts[path]) for n in lines
    ):
        raise ValueError('Invalid citation lines: ' + path)
    return dict(citation, quotations=[{'line': n, 'text': texts[path][n-1]} for n in lines])


def group_exposure(groups, seed_paths, used_containers=()):
    """Propagate deliberate use through blocks, not backwards through containers.

    Audit-only container inspection does not taint all its unrelated stories.
    An actually used whole container explicitly supplies the groups it contains.
    """
    exposed = {g['split_block'] for g in groups
               if any(m['source_path'] in seed_paths for m in g['members'])}
    by_id = {g['id']: g for g in groups}
    for group_id in used_containers:
        exposed.add(by_id[group_id]['split_block'])
    return exposed


def normalized_body(line):
    # Ignore speaker reassignment when proposing overlaps. Preserve originals in witnesses.
    body = re.sub(r'^[^:：]{1,30}[:：]', '', line)
    return ''.join(c for c in unicodedata.normalize('NFKC', body) if c.isalnum())


def overlap_hints(texts, minimum=20):
    """Sentence containment including wrapped long prompts; no independence inference."""
    normalized = {}
    for path, lines in texts.items():
        chunks = [normalized_body(line) for line in lines]
        positions = [i for i, chunk in enumerate(chunks, 1) for _ in chunk]
        normalized[path] = (''.join(chunks), positions, chunks)
    pairs = defaultdict(list)
    for a, (_, _, chunks) in normalized.items():
        for line_number, chunk in enumerate(chunks, 1):
            if len(chunk) < minimum:
                continue
            for b, (full, positions, _) in normalized.items():
                if a == b:
                    continue
                start = full.find(chunk)
                if start < 0:
                    continue
                key = tuple(sorted((a, b)))
                witness = {'source': a, 'line': line_number, 'contained_in': b,
                           'contained_lines': sorted(set(positions[start:start+len(chunk)])),
                           'normalized_chars': len(chunk)}
                pairs[key].append(witness)
    return [{'paths': list(pair), 'witness_count': len(witnesses),
             'witnesses': sorted(witnesses, key=lambda x: (-x['normalized_chars'], x['source'], x['line'])),
             'decision': 'overlap_hint_not_event_identity'}
            for pair, witnesses in sorted(pairs.items())]


def build(output, review_dir, root=ROOT):
    root = root.resolve()
    output = output.resolve()
    if not output.is_relative_to(root / 'runs'):
        raise ValueError('Audit output must be inside project runs/')
    if output.exists():
        raise ValueError('Output exists; use a new directory')
    inv_path = root / 'runs/stage_b_20260922/full_pool_01/inventory.json'
    review_path = review_dir / 'review.json'
    evidence_path = review_dir / 'evidence.json'
    review = read_json(review_path)
    evidence = read_json(evidence_path)
    if review['revision'] != REVISION or review['inventory_sha256'] != sha(inv_path.read_bytes()):
        raise ValueError('Pinned review/inventory mismatch')
    if review['formal_freeze'] or review['policy_or_judge_tuning_this_audit']:
        raise ValueError('This exporter only supports audit-only review')
    inventory = read_json(inv_path)
    sources = {r['path']: r for r in inventory}
    if len(sources) != len(inventory) or len(sources) != 528:
        raise ValueError('Incomplete or duplicate inventory')

    # Check the cached Git tree against its original manifest, then all local text bytes.
    old_root = root / 'data/raw/genshin_selection_20260921'
    source_manifest_path = old_root / 'sources.jsonl'
    old_manifest = [json.loads(line) for line in source_manifest_path.read_text(encoding='utf-8').splitlines()]
    manifest_by_path = {r['path']: r for r in old_manifest}
    tree_path = old_root / 'tree.json'
    if sha(tree_path.read_bytes()) != manifest_by_path['tree.json']['sha256']:
        raise ValueError('Pinned tree hash mismatch')
    tree = read_json(tree_path)
    if tree['sha'] != REVISION or tree.get('truncated'):
        raise ValueError('Wrong upstream revision/tree')
    entries = {r['path']: r for r in tree['tree']}
    expected = {p for p in entries if re.fullmatch(r'characters/(ayaka|hutao|zhongli)/texts/[^/]+\.txt', p)}
    if set(sources) != expected:
        raise ValueError('Inventory does not cover the pinned tree')
    for role in ('ayaka', 'hutao', 'zhongli'):
        p = f'characters/{role}/system_prompt.txt'
        sources[p] = dict(manifest_by_path[p], cache_path=(old_root / p).relative_to(root).as_posix(),
                          git_blob=entries[p]['sha'], role=role, kind='system_prompt')
    texts = {}
    for p, r in sources.items():
        if r['git_blob'] != entries[p]['sha'] or r['bytes'] != entries[p]['size']:
            raise ValueError('Inventory/tree mismatch: ' + p)
        texts[p] = verify_source(root, r)

    citations = []
    def resolve(c):
        value = resolve_citation(c, sources, texts)
        citations.append(value)
        return value

    groups = review['groups']
    if len({g['id'] for g in groups}) != len(groups):
        raise ValueError('Duplicate event group')
    membership = {}
    for g in groups:
        for member in g['members']:
            p = member['source_path']
            if p in membership:
                raise ValueError('Conflicting event membership: ' + p)
            membership[p] = g
            resolve(member)
    for container in review['containers']:
        resolve(container['source'])
        for contained in container['contains']:
            if contained['group'] not in {g['id'] for g in groups}:
                raise ValueError('Unknown contained event')
            resolve(dict(container['source'], lines=contained['lines']))
    for issue in review['issues']:
        for c in issue['sources']:
            resolve(c)
    for relation in review['relations']:
        for c in relation['sources']:
            resolve(c)
    for link in review['non_scene_links']:
        resolve({k:link[k] for k in ('source_path','source_sha256','lines')})
        resolve({'source_path':link['container_path'],
                 'source_sha256':link['container_sha256'],'lines':link['container_lines']})
    if len({e['id'] for e in evidence['items']}) != len(evidence['items']):
        raise ValueError('Duplicate evidence ID')
    for item in evidence['items']:
        for c in item['sources']:
            resolve(c)
        if item['human_verified'] or item['policy_attachment'] != 'none_pending_context_specific_review':
            raise ValueError('Unexpected approved evidence')

    index_path = root / 'docs/GENSHIN_STAGE_B_CONTEXTS.jsonl'
    prior = [json.loads(line) for line in index_path.read_text(encoding='utf-8').splitlines()]
    # Quarantined development sources also remain exposed; no rebranding as test.
    seed_paths = {r['source_path'] for r in prior if r['exposure'] == 'development_review'}
    evidence_paths = {c['source_path'] for item in evidence['items'] for c in item['sources']}
    exposed_blocks = group_exposure(groups, seed_paths | evidence_paths)
    reserves = []
    for cutoff in evidence['reserve_cutoffs']:
        c = cutoff['source']
        resolve(c)
        target = cutoff['target_line']
        p = c['source_path']
        if type(target) is not int or not max(c['lines']) < target <= len(texts[p]):
            raise ValueError('Future/invalid reserve history: ' + cutoff['id'])
        if not texts[p][target-1].startswith(('钟离:', '钟离：')):
            raise ValueError('Reserve target is not the target speaker')
        if membership[p]['split_block'] != cutoff['split_block'] or cutoff['split_block'] in exposed_blocks:
            raise ValueError('Reserve exposure/block conflict')
        reserves.append(dict(cutoff, visible_history=resolve_citation(c, sources, texts)['quotations'],
                             hidden_target_retained_in_raw_cache_only=True))

    hints = overlap_hints(texts)
    duplicate_sets = defaultdict(list)
    for r in inventory:
        duplicate_sets[r['sha256']].append(r['path'])
    duplicates = [v for v in duplicate_sets.values() if len(v)>1]
    for paths in duplicates:
        if len({membership[p]['split_block'] for p in paths if p in membership}) > 1:
            raise ValueError('Exact duplicate assigned to different blocks')
    issue_by_path = defaultdict(list)
    for issue in review['issues']:
        for c in issue['sources']:
            issue_by_path[c['source_path']].append(issue['id'])
    container_paths = {c['source']['source_path'] for c in review['containers']}
    correspondence_paths = {c['source_path'] for c in review['non_scene_links']}
    ledger = []
    for p, r in sources.items():
        g = membership.get(p)
        if g:
            disposition = 'linked_chain'
        elif p in container_paths or r['kind'] in {'profile_or_setting', 'system_prompt'}:
            disposition = 'excluded_profile_or_setting'
        elif p in correspondence_paths:
            disposition = 'correspondence_fragment_not_response_context'
        elif r['kind'] == 'voice_candidate' or p in evidence_paths or any(
            word in Path(p).name for word in ('爱好', '烦恼')
        ):
            disposition = 'voice_or_self_description_not_independent_event'
        else:
            disposition = 'unresolved_fragment'
        exposure = ('prior_development_review' if p in seed_paths else
                    'development_evidence_design' if p in evidence_paths else
                    'inherited_development_event' if g and g['split_block'] in exposed_blocks else
                    'prior_source_selection_review' if r['kind']=='system_prompt' else
                    'data_audit_only')
        ledger.append({'source_path':p, 'source_sha256':r['sha256'],'revision':REVISION,
                       'cache_path':r['cache_path'],'role':r['role'],
                       'previous_filename_kind':r['kind'],'disposition':disposition,
                       'group':g['id'] if g else None,'split_block':g['split_block'] if g else None,
                       'exposure':exposure,'use_this_run':'source_audit_no_generation_or_judge_tuning',
                       'issues':issue_by_path[p], 'assigned_split':None,'blind_test_approved':False})
    group_summary = [dict(id=g['id'], title=g['title'], split_block=g['split_block'],
                          files=len(g['members']), independent_events_verified=False,
                          blind_test_status='excluded_development_event' if g['split_block'] in exposed_blocks
                          else 'reserve_pending_provenance_temporal_and_exposure_acceptance') for g in groups]
    old_hints = read_json(root/'runs/stage_b_20260922/full_pool_01/overlap_candidates.json')
    old_pairs = {tuple(sorted(p)) for p in old_hints}
    new_pairs = {tuple(h['paths']) for h in hints}
    if not old_pairs <= new_pairs:
        raise ValueError('New overlap scan lost a previous candidate pair')
    summary = {'revision':REVISION, 'cache_files_verified':528, 'system_prompts_verified':3,
               'unique_text_contents':len(duplicate_sets), 'linked_files':len(membership),
               'event_chains':len(groups), 'split_blocks':len({g['split_block'] for g in groups}),
               'groups':group_summary, 'dispositions':dict(Counter(r['disposition'] for r in ledger)),
               'exposures':dict(Counter(r['exposure'] for r in ledger)),
               'overlap_minimum_body_chars':20,
               'overlap_pairs':len(hints), 'previous_overlap_pairs':len(old_pairs),
               'new_overlap_pairs':len(new_pairs-old_pairs),
               'cross_role_overlap_pairs':sum(sources[h['paths'][0]]['role']!=sources[h['paths'][1]]['role'] for h in hints),
               'evidence_items':len(evidence['items']), 'evidence_kinds':dict(Counter(i['kind'] for i in evidence['items'])),
               'reserve_cutoffs':len(reserves), 'reserve_blocks':len({r['split_block'] for r in reserves}),
               'formal_train_contexts':0,'formal_test_contexts':0,'verified_independent_capacity':None,
               'formal_freeze':False,'human_annotations_completed':0,'model_calls':0,
               'calibration_passed':False,
               'limitations':['链数、阻断块数、截点数都不是独立事件容量。',
                              '重叠扫描只发现候选；没有命中不能证明语义独立。',
                              '本轮助手数据审核不是人工标注或原作逐句认证。',
                              '三条保留链合成两块，尚未确认足够独立且可知证据完备的train/test。']}
    inputs = [inv_path, review_path, evidence_path, index_path, tree_path, source_manifest_path,
              Path(__file__).resolve()]
    output.mkdir(parents=True)
    write_json(output/'summary.json', summary)
    write_json(output/'source_ledger.json', ledger)
    write_json(output/'group_review.json', groups)
    write_json(output/'review_decisions.json', review)
    write_json(output/'citation_replay.local.json', citations)
    write_json(output/'evidence_review.local.json', evidence)
    write_json(output/'reserve_contexts.local.json', reserves)
    write_json(output/'overlap_witnesses.local.json', hints)
    write_json(output/'duplicates.json', duplicates)
    write_json(output/'exposure_review.json', {'prior_sources':sorted(seed_paths),
               'evidence_design_sources':sorted(evidence_paths),'excluded_blocks':sorted(exposed_blocks),
               'containers_not_deployed':review['containers'],
               'rule':'Audit-only reading is not tuning. Whole-container later use taints all contained groups; never use these containers as profiles.'})
    write_json(output/'audit_manifest.json', {
        'kind':'audit_integrity_only_not_formal_freeze',
        'inputs':{p.relative_to(root).as_posix():sha(p.read_bytes()) for p in inputs},
        'files':{p.name:sha(p.read_bytes()) for p in sorted(output.iterdir())},
        'source_hashes':{p:r['sha256'] for p,r in sources.items()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--review-dir',type=Path,default=ROOT/'docs/stage_b_data_audit_v01')
    args=parser.parse_args()
    print(json.dumps(build(args.output,args.review_dir),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
