import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
import install_once as I

class InstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.stage=self.root/'stage';self.stage.mkdir()
        self.journal=self.root/'journal.jsonl'
    def row(self,name,data):
        h=hashlib.sha256(data).hexdigest();p=self.stage/h
        if not p.exists():p.write_bytes(data)
        return {'path':name,'blob_sha256':h,'bytes':len(data),'roles':['fixture']}
    def manifest(self,rows):
        blobs={r['blob_sha256']:r['bytes'] for r in rows}
        return {'schema':'openh3.content_addressed_missing_dependency_archive.v1','files':rows,
            'path_count':len(rows),'blob_count':len(blobs),'unique_source_bytes':sum(blobs.values())}
    def archive(self,manifest,extra=None,wrong_blob=False):
        raw=(json.dumps(manifest)+'\n').encode();path=self.root/'input.tar.gz'
        with tarfile.open(path,'w:gz',compresslevel=1) as t:
            for h,n in {r['blob_sha256']:r['bytes'] for r in manifest['files']}.items():
                data=(self.stage/h).read_bytes();data=b'X'*len(data) if wrong_blob else data
                ti=tarfile.TarInfo('blobs/'+h);ti.size=n;t.addfile(ti,io.BytesIO(data))
            ti=tarfile.TarInfo('manifest.json');ti.size=len(raw);t.addfile(ti,io.BytesIO(raw))
            if extra is not None:t.addfile(extra,io.BytesIO(b''))
        return path,raw
    def test_actual_archive_SHA_blob_and_manifest(self):
        m=self.manifest([self.row('new/file',b'abc')]);path,raw=self.archive(m)
        _,blobs=I.paths_and_blobs(m);destination=self.root/'freshstage';destination.mkdir()
        result=I.verified_archive(path,I.sha(path),raw,blobs,destination)
        self.assertEqual(result['members_verified'],2)
        self.assertEqual((destination/m['files'][0]['blob_sha256']).read_bytes(),b'abc')
        with self.assertRaisesRegex(ValueError,'archive SHA'):I.verified_archive(path,'0'*64,raw,blobs)
    def test_actual_member_hash_mismatch_blocks_commit(self):
        m=self.manifest([self.row('new/file',b'abc')]);path,raw=self.archive(m,wrong_blob=True)
        with self.assertRaisesRegex(ValueError,'blob SHA'):I.verified_archive(path,I.sha(path),raw,I.paths_and_blobs(m)[1])
        self.assertFalse((self.root/'new').exists())
    def test_traversal_and_duplicate_destination_rejected(self):
        for name in ('../escape','/absolute','a//b','a/../b'):
            with self.subTest(name=name),self.assertRaises(ValueError):I.paths_and_blobs(self.manifest([self.row(name,b'x')]))
        row=self.row('a',b'x')
        with self.assertRaises(ValueError):I.paths_and_blobs(self.manifest([row,row]))
    def test_link_member_rejected_without_extract(self):
        m=self.manifest([self.row('new/file',b'abc')]);ti=tarfile.TarInfo('unexpected');ti.type=tarfile.SYMTYPE;ti.linkname='/tmp/outside'
        path,raw=self.archive(m,extra=ti)
        with self.assertRaisesRegex(ValueError,'unsafe'):I.verified_archive(path,I.sha(path),raw,I.paths_and_blobs(m)[1])
    def test_existing_conflict_preserved_no_partial_install(self):
        (self.root/'exists').write_bytes(b'OLD');inode=(self.root/'exists').stat().st_ino
        paths={r['path']:r for r in [self.row('first/new',b'new'),self.row('exists',b'BAD')]}
        with self.assertRaises(ValueError):I.commit(self.root,paths,self.stage,self.journal)
        self.assertEqual((self.root/'exists').read_bytes(),b'OLD');self.assertEqual((self.root/'exists').stat().st_ino,inode)
        self.assertFalse((self.root/'first').exists())
    def test_identical_existing_skip_and_missing_install(self):
        (self.root/'same').write_bytes(b'same');inode=(self.root/'same').stat().st_ino
        paths={r['path']:r for r in [self.row('same',b'same'),self.row('new/sub/file',b'new')]}
        r=I.commit(self.root,paths,self.stage,self.journal)
        self.assertEqual(r['existing_identical_preserved'],1);self.assertEqual(r['installed_paths'],1)
        self.assertEqual((self.root/'same').stat().st_ino,inode);self.assertEqual((self.root/'new/sub/file').read_bytes(),b'new')
    def test_parent_symlink_never_followed(self):
        other=self.root/'outside';other.mkdir();os.symlink(other,self.root/'link')
        row=self.row('link/file',b'new')
        with self.assertRaises(OSError):I.commit(self.root,{row['path']:row},self.stage,self.journal)
        self.assertFalse((other/'file').exists())
    def test_duplicate_blob_destinations_have_independent_inodes(self):
        rows=[self.row('snapshot_one/file',b'same'),self.row('snapshot_two/file',b'same')]
        result=I.commit(self.root,{r['path']:r for r in rows},self.stage,self.journal)
        one=self.root/rows[0]['path'];two=self.root/rows[1]['path'];blob=self.stage/rows[0]['blob_sha256']
        self.assertTrue(result['new_destination_inodes_independent'])
        self.assertEqual(len({p.stat().st_ino for p in (one,two,blob)}),3)
        os.chmod(one,0o644);one.write_bytes(b'changed')
        self.assertEqual(two.read_bytes(),b'same');self.assertEqual(blob.read_bytes(),b'same')
    def test_commit_race_rolls_back_only_own_files(self):
        paths={r['path']:r for r in [self.row('first/new',b'one'),self.row('race',b'two')]}
        def race(name):
            if name=='race':(self.root/'race').write_bytes(b'EXTERNAL')
        with self.assertRaises(FileExistsError):I.commit(self.root,paths,self.stage,self.journal,before_link=race)
        self.assertFalse((self.root/'first').exists());self.assertEqual((self.root/'race').read_bytes(),b'EXTERNAL')
        self.assertIn('rollback_owned_file',self.journal.read_text())

if __name__=='__main__':unittest.main(verbosity=2)
