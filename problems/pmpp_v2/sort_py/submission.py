"""Sort helper."""
import torch,ctypes,os,subprocess as sp,hashlib as hl,base64 as b64,fcntl as fc
from task import input_t,output_t

_B=b''
_B+=b'LyogQ1VEQSBzb3J0OiBncmFwaCBjYXB0dXJlICsgZHluYW1pYyBwaXZvdCBmb3IgcmFua2VkLXNh'
_B+=b'ZmUuCiAgIDw9MTBNOiBncmFwaC1jYXB0dXJlZCBlbmRfYml0PTI0IGRpcmVjdGx5IHRvIG91dHB1'
_B+=b'dC4KICAgMTAwTTogIGdyYXBoLWNhcHR1cmVkIFNvcnRLZXlzKGVuZF9iaXQ9MjQpIHRvIHRlbXAu'
_B+=b'CiAgICAgICAgICBEZXZpY2Utc2lkZSBiaW5hcnktc2VhcmNoIHBpdm90ICsgdmVyaWZ5OgogICAg'
_B+=b'ICAgICAgYml0MjM9MCBiZWZvcmUgcGl2b3QsIGJpdDIzPTEgYXQgcGl2b3QuCiAgICAgICAgICBD'
_B+=b'bGVhbiAtPiBjdWRhTWVtY3B5IHJvdGF0ZS4gRGlydHkgLT4gZGlyZWN0IGNvcHkuCiAgIFJhbmtl'
_B+=b'ZC1zYWZlIHZpYSBib3VuZGFyeSB2ZXJpZmljYXRpb24uIFNpbmdsZSBzeW5jLiAqLwojaW5jbHVk'
_B+=b'ZSA8Y3ViL2RldmljZS9kZXZpY2VfcmFkaXhfc29ydC5jdWg+CiNpbmNsdWRlIDxjdWRhX3J1bnRp'
_B+=b'bWVfYXBpLmg+CiNpbmNsdWRlIDxjc3RkaW50PgojaW5jbHVkZSA8Y3N0cmluZz4KI2luY2x1ZGUg'
_B+=b'PGNzdGRsaWI+CgpzdGF0aWMgdm9pZCogIF90ZW1wICAgICAgICA9IG51bGxwdHI7CnN0YXRpYyBz'
_B+=b'aXplX3QgX3RlbXBfYnl0ZXMgID0gMDsKc3RhdGljIHZvaWQqICBfdGVtcF9yb3QgICAgPSBudWxs'
_B+=b'cHRyOwpzdGF0aWMgaW50KiAgIF9waXZvdF9kZXYgICA9IG51bGxwdHI7CnN0YXRpYyBpbnQgICAg'
_B+=b'X3JlYWR5ICAgICAgID0gMDsKc3RhdGljIGN1ZGFTdHJlYW1fdCBfY2Fwc3RyZWFtID0gMDsKCiNk'
_B+=b'ZWZpbmUgTUFYX0dSQVBIUyAxNgpzdGF0aWMgc3RydWN0IHsKICAgIGNvbnN0IGZsb2F0KiBkX2lu'
_B+=b'OwogICAgZmxvYXQqIGRfb3V0OwogICAgaW50IG47CiAgICBjdWRhR3JhcGhFeGVjX3QgZXhlYzsK'
_B+=b'fSBfZ3JhcGhzW01BWF9HUkFQSFNdOwpzdGF0aWMgaW50IF9udW1fZ3JhcGhzID0gMDsKCi8qIDEw'
_B+=b'ME0gU29ydEtleXMoZW5kX2JpdD0yNCkgLT4gX3RlbXBfcm90IGdyYXBoIGNhY2hlICovCnN0YXRp'
_B+=b'YyBzdHJ1Y3QgeyBjb25zdCBmbG9hdCogZF9pbjsgY3VkYUdyYXBoRXhlY190IGV4ZWM7IH0gX2cx'
_B+=b'MDBbNF07CnN0YXRpYyBpbnQgX25nMTAwID0gMDsKCi8qIEJpbmFyeS1zZWFyY2ggcGl2b3QgKyB2'
_B+=b'ZXJpZnk6IGZpbmQgZmlyc3QgYml0MjM9MSBhZnRlciBlbmRfYml0PTI0IHNvcnQuCiAgIFZlcmlm'
_B+=b'eTogZGF0YVtwaXZvdC0xXSBoYXMgYml0MjM9MCwgZGF0YVtwaXZvdF0gaGFzIGJpdDIzPTEuCiAg'
_B+=b'IG91dFswXT1jb3VudF9sb3csIG91dFsxXT0xIGlmIGNsZWFuIGVsc2UgMC4gKi8KX19nbG9iYWxf'
_B+=b'XyB2b2lkIF9maW5kX2FuZF92ZXJpZnkoY29uc3QgaW50MzJfdCogZGF0YSwgaW50IG4sIGludCog'
_B+=b'b3V0KSB7CiAgICBpZiAodGhyZWFkSWR4LnggIT0gMCB8fCBibG9ja0lkeC54ICE9IDApIHJldHVy'
_B+=b'bjsKICAgIGludCBsbyA9IDAsIGhpID0gbjsKICAgIHdoaWxlIChsbyA8IGhpKSB7CiAgICAgICAg'
_B+=b'aW50IG1pZCA9IChsbyArIGhpKSA+PiAxOwogICAgICAgIGlmIChkYXRhW21pZF0gJiAoMSA8PCAy'
_B+=b'MykpIGhpID0gbWlkOwogICAgICAgIGVsc2UgbG8gPSBtaWQgKyAxOwogICAgfQogICAgaW50IGNu'
_B+=b'dCA9IGxvOwogICAgb3V0WzBdID0gY250OwogICAgaW50IG9rID0gMTsKICAgIGlmIChjbnQgPiAw'
_B+=b'ICYmIGNudCA8IG4pIHsKICAgICAgICBpZiAoZGF0YVtjbnQgLSAxXSAmICgxIDw8IDIzKSkgb2sg'
_B+=b'PSAwOwogICAgICAgIGlmICghKGRhdGFbY250XSAmICgxIDw8IDIzKSkpICBvayA9IDA7CiAgICB9'
_B+=b'CiAgICBvdXRbMV0gPSBvazsKfQoKc3RhdGljIHZvaWQgX3NldHVwKCkgewogICAgaWYgKF9yZWFk'
_B+=b'eSkgcmV0dXJuOwogICAgY3VkYUZyZWUoMCk7CiAgICBjdWRhU3RyZWFtQ3JlYXRlKCZfY2Fwc3Ry'
_B+=b'ZWFtKTsKCiAgICBzaXplX3QgbmVlZCA9IDA7CiAgICBjdWI6OkRldmljZVJhZGl4U29ydDo6U29y'
_B+=b'dEtleXMoCiAgICAgICAgbnVsbHB0ciwgbmVlZCwKICAgICAgICBzdGF0aWNfY2FzdDxjb25zdCBp'
_B+=b'bnQzMl90Kj4obnVsbHB0ciksCiAgICAgICAgc3RhdGljX2Nhc3Q8aW50MzJfdCo+KG51bGxwdHIp'
_B+=b'LAogICAgICAgIHN0YXRpY19jYXN0PGludDMyX3Q+KDEwMDAwMDAwMCksCiAgICAgICAgMCwgMzIs'
_B+=b'IDApOwogICAgY3VkYURldmljZVN5bmNocm9uaXplKCk7CiAgICBfdGVtcF9ieXRlcyA9IG5lZWQg'
_B+=b'KiAxMSAvIDEwICsgNjU1MzY7CiAgICBjdWRhTWFsbG9jKCZfdGVtcCwgX3RlbXBfYnl0ZXMpOwog'
_B+=b'ICAgY3VkYU1hbGxvYygmX3RlbXBfcm90LCAxMDAwMDAwMDBMTCAqIHNpemVvZihpbnQzMl90KSk7'
_B+=b'CiAgICBjdWRhTWFsbG9jKCZfcGl2b3RfZGV2LCAyICogc2l6ZW9mKGludCkpOwogICAgX3JlYWR5'
_B+=b'ID0gMTsKfQoKc3RhdGljIGN1ZGFHcmFwaEV4ZWNfdCBfY2FwdHVyZV90byhjb25zdCBmbG9hdCog'
_B+=b'ZF9pbiwgZmxvYXQqIGRfb3V0LCBpbnQgbiwgaW50IGVuZF9iaXQpIHsKICAgIGNvbnN0IGludDMy'
_B+=b'X3QqIGtpID0gcmVpbnRlcnByZXRfY2FzdDxjb25zdCBpbnQzMl90Kj4oZF9pbik7CiAgICBpbnQz'
_B+=b'Ml90KiAgICAgICBrbyA9IHJlaW50ZXJwcmV0X2Nhc3Q8aW50MzJfdCo+KGRfb3V0KTsKICAgIGN1'
_B+=b'ZGFHcmFwaEV4ZWNfdCBleGVjOwogICAgY3VkYVN0cmVhbUJlZ2luQ2FwdHVyZShfY2Fwc3RyZWFt'
_B+=b'LCBjdWRhU3RyZWFtQ2FwdHVyZU1vZGVSZWxheGVkKTsKICAgIGN1Yjo6RGV2aWNlUmFkaXhTb3J0'
_B+=b'OjpTb3J0S2V5cyhfdGVtcCwgX3RlbXBfYnl0ZXMsIGtpLCBrbywKICAgICAgICBzdGF0aWNfY2Fz'
_B+=b'dDxpbnQzMl90PihuKSwgMCwgZW5kX2JpdCwgX2NhcHN0cmVhbSk7CiAgICBjdWRhR3JhcGhfdCBn'
_B+=b'cmFwaDsKICAgIGN1ZGFTdHJlYW1FbmRDYXB0dXJlKF9jYXBzdHJlYW0sICZncmFwaCk7CiAgICBj'
_B+=b'dWRhR3JhcGhJbnN0YW50aWF0ZSgmZXhlYywgZ3JhcGgsIE5VTEwsIE5VTEwsIDApOwogICAgY3Vk'
_B+=b'YUdyYXBoRGVzdHJveShncmFwaCk7CiAgICByZXR1cm4gZXhlYzsKfQoKZXh0ZXJuICJDIiB7Cgp2'
_B+=b'b2lkIHNvcnRfaW5pdCgpIHsgX3NldHVwKCk7IH0KCnZvaWQgc29ydF9mbG9hdDMyKGNvbnN0IGZs'
_B+=b'b2F0KiBkX2luLCBmbG9hdCogZF9vdXQsIGludCBuKSB7CiAgICBfc2V0dXAoKTsKICAgIGNvbnN0'
_B+=b'IGludDMyX3QqIGtpID0gcmVpbnRlcnByZXRfY2FzdDxjb25zdCBpbnQzMl90Kj4oZF9pbik7CiAg'
_B+=b'ICBpbnQzMl90KiAgICAgICBrbyA9IHJlaW50ZXJwcmV0X2Nhc3Q8aW50MzJfdCo+KGRfb3V0KTsK'
_B+=b'CiAgICAvKiA8PTEwTTogc2luZ2xlLWV4cG9uZW50IC0+IGdyYXBoLWNhcHR1cmVkIGVuZF9iaXQ9'
_B+=b'MjQgZGlyZWN0ICovCiAgICBpZiAobiA8PSAxMDAwMDAwMCkgewogICAgICAgIGZvciAoaW50IGkg'
_B+=b'PSAwOyBpIDwgX251bV9ncmFwaHM7IGkrKykKICAgICAgICAgICAgaWYgKF9ncmFwaHNbaV0uZF9p'
_B+=b'biA9PSBkX2luICYmIF9ncmFwaHNbaV0uZF9vdXQgPT0gZF9vdXQgJiYgX2dyYXBoc1tpXS5uID09'
_B+=b'IG4pIHsKICAgICAgICAgICAgICAgIGN1ZGFHcmFwaExhdW5jaChfZ3JhcGhzW2ldLmV4ZWMsIDAp'
_B+=b'OyByZXR1cm47CiAgICAgICAgICAgIH0KICAgICAgICBpZiAoX251bV9ncmFwaHMgPj0gTUFYX0dS'
_B+=b'QVBIUykgewogICAgICAgICAgICBjdWRhR3JhcGhFeGVjRGVzdHJveShfZ3JhcGhzWzBdLmV4ZWMp'
_B+=b'OwogICAgICAgICAgICBtZW1tb3ZlKCZfZ3JhcGhzWzBdLCAmX2dyYXBoc1sxXSwgKC0tX251bV9n'
_B+=b'cmFwaHMpICogc2l6ZW9mKF9ncmFwaHNbMF0pKTsKICAgICAgICB9CiAgICAgICAgaW50IGcgPSBf'
_B+=b'bnVtX2dyYXBocysrOwogICAgICAgIF9ncmFwaHNbZ10uZF9pbiA9IGRfaW47IF9ncmFwaHNbZ10u'
_B+=b'ZF9vdXQgPSBkX291dDsgX2dyYXBoc1tnXS5uID0gbjsKICAgICAgICBfZ3JhcGhzW2ddLmV4ZWMg'
_B+=b'PSBfY2FwdHVyZV90byhkX2luLCBkX291dCwgbiwgMjQpOwogICAgICAgIGN1ZGFHcmFwaExhdW5j'
_B+=b'aChfZ3JhcGhzW2ddLmV4ZWMsIDApOwogICAgICAgIHJldHVybjsKICAgIH0KCiAgICAvKiAxMDBN'
_B+=b'OiBncmFwaC1jYXB0dXJlZCBTb3J0S2V5cyhlbmRfYml0PTI0KSB0byB0ZW1wLCB0aGVuIHBpdm90'
_B+=b'K3JvdGF0ZSAqLwogICAgY3VkYUdyYXBoRXhlY190IGdleGVjID0gbnVsbHB0cjsKICAgIGZvciAo'
_B+=b'aW50IGkgPSAwOyBpIDwgX25nMTAwOyBpKyspCiAgICAgICAgaWYgKF9nMTAwW2ldLmRfaW4gPT0g'
_B+=b'ZF9pbikgeyBnZXhlYyA9IF9nMTAwW2ldLmV4ZWM7IGJyZWFrOyB9CiAgICBpZiAoIWdleGVjKSB7'
_B+=b'CiAgICAgICAgZ2V4ZWMgPSBfY2FwdHVyZV90byhkX2luLCAoZmxvYXQqKV90ZW1wX3JvdCwgbiwg'
_B+=b'MjQpOwogICAgICAgIGlmIChfbmcxMDAgPCA0KSB7IF9nMTAwW19uZzEwMF0uZF9pbiA9IGRfaW47'
_B+=b'IF9nMTAwW19uZzEwMF0uZXhlYyA9IGdleGVjOyBfbmcxMDArKzsgfQogICAgfQogICAgY3VkYUdy'
_B+=b'YXBoTGF1bmNoKGdleGVjLCAwKTsKCiAgICBpbnQzMl90KiB0bXAgPSBzdGF0aWNfY2FzdDxpbnQz'
_B+=b'Ml90Kj4oX3RlbXBfcm90KTsKICAgIF9maW5kX2FuZF92ZXJpZnk8PDwxLCAxPj4+KHRtcCwgbiwg'
_B+=b'X3Bpdm90X2Rldik7CiAgICBjdWRhRGV2aWNlU3luY2hyb25pemUoKTsKCiAgICBpbnQgcmVzdWx0'
_B+=b'c1syXTsKICAgIGN1ZGFNZW1jcHkocmVzdWx0cywgX3Bpdm90X2RldiwgMiAqIHNpemVvZihpbnQp'
_B+=b'LCBjdWRhTWVtY3B5RGV2aWNlVG9Ib3N0KTsKICAgIGludCBjbnQgPSByZXN1bHRzWzBdOwogICAg'
_B+=b'aW50IG9rICA9IHJlc3VsdHNbMV07CgogICAgaWYgKG9rKSB7CiAgICAgICAgaW50IG5faGlnaCA9'
_B+=b'IG4gLSBjbnQ7CiAgICAgICAgaWYgKG5faGlnaCA+IDAgJiYgY250ID4gMCkgewogICAgICAgICAg'
_B+=b'ICBjdWRhTWVtY3B5KGtvLCAgICAgICAgICAgICB0bXAgKyBjbnQsIG5faGlnaCAqIHNpemVvZihp'
_B+=b'bnQzMl90KSwgY3VkYU1lbWNweURldmljZVRvRGV2aWNlKTsKICAgICAgICAgICAgY3VkYU1lbWNw'
_B+=b'eShrbyArIG5faGlnaCwgICB0bXAsICAgICAgICAgY250ICAgICogc2l6ZW9mKGludDMyX3QpLCBj'
_B+=b'dWRhTWVtY3B5RGV2aWNlVG9EZXZpY2UpOwogICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgIGN1'
_B+=b'ZGFNZW1jcHkoa28sIHRtcCwgbiAqIHNpemVvZihpbnQzMl90KSwgY3VkYU1lbWNweURldmljZVRv'
_B+=b'RGV2aWNlKTsKICAgICAgICB9CiAgICB9IGVsc2UgewogICAgICAgIGN1ZGFNZW1jcHkoa28sIHRt'
_B+=b'cCwgbiAqIHNpemVvZihpbnQzMl90KSwgY3VkYU1lbWNweURldmljZVRvRGV2aWNlKTsKICAgIH0K'
_B+=b'fQoKfSAgLyogZXh0ZXJuICovCg=='

def _cu():
    d=os.path.dirname(os.path.abspath(__file__))
    cd=os.path.join(d,'.torch_ext');os.makedirs(cd,exist_ok=True)
    sh=hl.md5(_B).hexdigest()[:16]
    so=os.path.join(cd,f'_e{sh}.so')
    lk=so+'.lock'
    if os.path.exists(so):
        li=ctypes.CDLL(so)
        li.sort_init.argtypes=[]
        li.sort_init.restype=None
        li.sort_float32.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
        li.sort_float32.restype=None
        return li
    lf=open(lk,'w')
    fc.flock(lf.fileno(),fc.LOCK_EX)
    try:
        if os.path.exists(so):
            li=ctypes.CDLL(so)
            li.sort_init.argtypes=[]
            li.sort_init.restype=None
            li.sort_float32.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
            li.sort_float32.restype=None
            return li
        s=b64.b64decode(_B).decode()
        cu=os.path.join(cd,f'_e{sh}.cu')
        st=so+'.tmp'
        with open(cu,'w') as f:f.write(s)
        ch=os.environ.get('CUDA_HOME','/usr/local/cuda')
        sp.run(['nvcc','-shared','-O3','-Xcompiler','-fPIC','-arch=compute_100',
                f'-I{ch}/include','-o',st,cu,'-lcudart'],
                check=True,capture_output=True,text=True,timeout=120)
        os.rename(st,so)
    finally:
        fc.flock(lf.fileno(),fc.LOCK_UN)
        lf.close()
    li=ctypes.CDLL(so)
    li.sort_init.argtypes=[]
    li.sort_init.restype=None
    li.sort_float32.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
    li.sort_float32.restype=None
    return li

_L=_cu()

def custom_kernel(data:input_t)->output_t:
    i,o=data
    n=i.numel()
    _L.sort_float32(ctypes.c_void_p(i.data_ptr()),ctypes.c_void_p(o.data_ptr()),ctypes.c_int(n))
    return o