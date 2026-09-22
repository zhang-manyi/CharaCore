"""Audit integrity/contamination checks; synthetic fixtures are not character evidence."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.audit_genshin_sources import (
    group_exposure, overlap_hints, resolve_citation, sha, verify_source,
)


class SourceAuditTests(unittest.TestCase):
    def test_source_bytes_cannot_change_silently(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            data='甲：材料。\n'.encode('utf-8')
            (root/'source.txt').write_bytes(data)
            row={'path':'upstream.txt','cache_path':'source.txt','sha256':sha(data),
                 'bytes':len(data),'git_blob':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()}
            self.assertEqual(verify_source(root,row),['甲：材料。'])
            (root/'source.txt').write_bytes(data+b'x')
            with self.assertRaisesRegex(ValueError,'integrity'):
                verify_source(root,row)

    def test_citation_rejects_wrong_hash_and_lines(self):
        sources={'x':{'sha256':'correct'}}
        texts={'x':['one','two']}
        good={'source_path':'x','source_sha256':'correct','lines':[2]}
        self.assertEqual(resolve_citation(good,sources,texts)['quotations'][0]['text'],'two')
        for bad in [dict(good,source_sha256='wrong'),dict(good,lines=[3]),
                    dict(good,lines=[True]),dict(good,lines=[2,1]),dict(good,lines=[])]:
            with self.assertRaises(ValueError):
                resolve_citation(bad,sources,texts)

    def test_dev_exposure_propagates_to_recap_block_only(self):
        groups=[{'id':'original','split_block':'history','members':[{'source_path':'a'}]},
                {'id':'recap','split_block':'history','members':[{'source_path':'b'}]},
                {'id':'tea','split_block':'tea','members':[{'source_path':'c'}]}]
        self.assertEqual(group_exposure(groups,{'b'}),{'history'})
        # Merely inspecting the container is not equivalent to using its entire contents.
        self.assertEqual(group_exposure(groups,{'container'}),set())
        self.assertEqual(group_exposure(groups,{'container'},['original','tea']),{'history','tea'})

    def test_containment_finds_reassigned_speaker_and_wrapped_prompt(self):
        texts={'scene':['派蒙：今天我们先到商会寻找丢失的货物随后回到店铺完成之前委托的事情。'],
               'prompt':['旅行者：今天我们先到商会寻找丢失的货物', '随后回到店铺完成之前委托的事情。'],
               'unrelated':['派蒙：商会。']}
        hints=overlap_hints(texts)
        self.assertEqual([h['paths'] for h in hints],[['prompt','scene']])
        self.assertEqual(hints[0]['witnesses'][0]['contained_lines'],[1,2])
        self.assertEqual(hints[0]['decision'],'overlap_hint_not_event_identity')


if __name__=='__main__':
    unittest.main()
