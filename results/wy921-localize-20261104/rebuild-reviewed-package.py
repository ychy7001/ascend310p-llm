"""Repack only reviewed text files with deterministic, identity-free metadata."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
prior = json.loads((root / 'repro-manifest.json').read_text())
package = root / 'upstream-gdn-wy921-runtime-repro'
files = sorted(x for x in package.iterdir() if x.is_file() and x.name != 'SHA256SUMS')
assert all(x.suffix in ('.py', '.md', '.json') for x in files)
(package / 'SHA256SUMS').write_text(''.join(hashlib.sha256(x.read_bytes()).hexdigest() + '  ' + x.name + '\n' for x in files))
files = sorted(files + [package / 'SHA256SUMS'])
archive = root / (package.name + '.tar.gz')
with archive.open('wb') as target, gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0) as compressed:
    with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as tar:
        folder = tarfile.TarInfo(package.name + '/')
        folder.type = tarfile.DIRTYPE
        folder.mode = 0o755
        folder.uid = folder.gid = folder.mtime = 0
        folder.uname = folder.gname = ''
        tar.addfile(folder)
        for path in files:
            data = path.read_bytes()
            info = tarfile.TarInfo(package.name + '/' + path.name)
            info.size = len(data)
            info.mode = 0o644
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ''
            tar.addfile(info, io.BytesIO(data))
manifest = {
    'path': str(archive.relative_to(root.parents[1])),
    'size_bytes': archive.stat().st_size,
    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'unpacked_path': str(package.relative_to(root.parents[1])),
    'contains_weights': False, 'contains_prompts': False, 'contains_binaries': False,
    'inputs': 'one 64-token, one-head activation snapshot in lossless decimal JSON',
    'files': [{'path': x.name, 'size_bytes': x.stat().st_size, 'sha256': hashlib.sha256(x.read_bytes()).hexdigest()} for x in files],
    'published': prior.get('published', False),
    'requires_build': 'PR17640 18163f6b1d208ab3b4fb06e9752225611c37f40e; CANN 9.2.1; torch 2.10.0; torch-npu 2.10.0.post7; no private binaries',
    'archive_metadata': {'uid': 0, 'gid': 0, 'uname': '', 'gname': '', 'mtime': 0, 'format': 'ustar', 'gzip_filename': ''},
}
for key in ('gist_url', 'comment_url', 'published_at_utc'):
    if key in prior:
        manifest[key] = prior[key]
(root / 'repro-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
a = json.loads((root / 'adjudication.json').read_text())
a['repro'] = manifest
for device in a['B2']['synthetic_devices']:
    device['loaded_apis'] = [x.replace('hyperqwen_gdn_pair921_transformer', '<paired-vendor>') for x in device['loaded_apis']]
(root / 'adjudication.json').write_text(json.dumps(a, ensure_ascii=False, indent=2) + '\n')
readme = root / 'README.md'
text = readme.read_text().replace('`qwen36-image.service`', '`<local-image-service>`')
start = text.index('复现包路径、大小、逐文件SHA见')
end = text.index('\n\n最小例', start)
text = text[:start] + f'复现包路径、大小、逐文件SHA见 `repro-manifest.json`。独立审查后重新打包：`upstream-gdn-wy921-runtime-repro.tar.gz` **{archive.stat().st_size:,}字节**；无权重、prompt、tokenizer或二进制，使用可无损还原的数值JSON输入与设备记录输出，含英文README、CPU参考/测试/校验、固定PR绑定构建及对照脚本。tar身份与时间元数据已清除。需使用CANN9.2.1和固定PR源码重新构建设备部分。发布状态见本文末尾审查记录。' + text[end:]
readme.write_text(text)
print(json.dumps({'files': len(files), 'size_bytes': manifest['size_bytes'], 'sha256': manifest['sha256']}, indent=2))
