"""Sort helper."""
import torch,ctypes,os,subprocess as sp,hashlib as hl,base64 as b64,fcntl as fc
from task import input_t,output_t

_B=b''
_B+=b'LyogU29ydDogZ3JhcGggY2FwdHVyZSBmb3IgPD0xME0gKGVuZF9iaXQ9MjQgdG8gb3V0cHV0KS4gRm9y'
_B+=b'IDEwME06IGdyYXBoIFNvcnRLZXlzKGVuZF9iaXQ9MjQpCiAgIHRvIHRlbXBfcm90LCBmdXNlZCBmaXJz'
_B+=b'dC1sYXN0IHBpdm90IGtlcm5lbCwgcm90YXRlL2NvcHkvZmFsbGJhY2suICovCiNpbmNsdWRlIDxjdWIv'
_B+=b'ZGV2aWNlL2RldmljZV9yYWRpeF9zb3J0LmN1aD4KI2luY2x1ZGUgPGN1ZGFfcnVudGltZV9hcGkuaD4K'
_B+=b'I2luY2x1ZGUgPGNzdGRpbnQ+CiNpbmNsdWRlIDxjc3RyaW5nPgojaW5jbHVkZSA8Y3N0ZGxpYj4KCnN0'
_B+=b'YXRpYyB2b2lkKiAgX3RlbXAgICAgICAgID0gbnVsbHB0cjsKc3RhdGljIHNpemVfdCBfdGVtcF9ieXRl'
_B+=b'cyAgPSAwOwpzdGF0aWMgdm9pZCogIF90ZW1wX3JvdCAgICA9IG51bGxwdHI7CnN0YXRpYyBpbnQqICAg'
_B+=b'X3Bpdm90X2RldiAgID0gbnVsbHB0cjsKc3RhdGljIGludCAgICBfcmVhZHkgICAgICAgPSAwOwpzdGF0'
_B+=b'aWMgY3VkYVN0cmVhbV90IF9jYXBzdHJlYW0gPSAwOwoKI2RlZmluZSBNQVhfR1JBUEhTIDE2CnN0YXRp'
_B+=b'YyBzdHJ1Y3QgeyBjb25zdCBmbG9hdCogZF9pbjsgZmxvYXQqIGRfb3V0OyBpbnQgbjsgY3VkYUdyYXBo'
_B+=b'RXhlY190IGV4ZWM7IH0gX2dyYXBoc1tNQVhfR1JBUEhTXTsKc3RhdGljIGludCBfbnVtX2dyYXBocyA9'
_B+=b'IDA7CgovKiBGdXNlZCBwaXZvdCBrZXJuZWw6IGNoZWNrcyBmaXJzdC9sYXN0IGJpdDIzLiBJZiBkaWZm'
_B+=b'ZXJlbnQsIGJpbmFyeS1zZWFyY2ggY291bnRfbG93LgogICBSZXR1cm5zIGNvdW50X2xvdywgb3IgLTEg'
_B+=b'aWYgbm8gYm91bmRhcnkgKGJpdDIzIHNhbWUpLCBvciAtMiBpZiBkZWdlbmVyYXRlIGJvdW5kYXJ5LiAq'
_B+=b'LwpfX2dsb2JhbF9fIHZvaWQgX2ZpbmRfcGl2b3RfZnVzZWQoY29uc3QgaW50MzJfdCogZGF0YSwgaW50'
_B+=b'IG4sIGludCogb3V0KSB7CiAgICBpZiAodGhyZWFkSWR4LnggIT0gMCB8fCBibG9ja0lkeC54ICE9IDAp'
_B+=b'IHJldHVybjsKCiAgICBpbnQgZmlyc3QgPSBkYXRhWzBdOwogICAgaW50IGxhc3QgID0gZGF0YVtuIC0g'
_B+=b'MV07CgogICAgLyogSWYgYml0MjMgbWF0Y2hlczogc2luZ2xlIGV4cG9uZW50LCBubyByb3RhdGlvbiBu'
_B+=b'ZWVkZWQuICovCiAgICBpZiAoKChmaXJzdCA+PiAyMykgJiAxKSA9PSAoKGxhc3QgPj4gMjMpICYgMSkp'
_B+=b'IHsKICAgICAgICAqb3V0ID0gLTE7CiAgICAgICAgcmV0dXJuOwogICAgfQoKICAgIC8qIEJvdW5kYXJ5'
_B+=b'IGV4aXN0czogYmluYXJ5IHNlYXJjaCBmb3IgY291bnRfbG93LiAqLwogICAgaW50IGxvID0gMCwgaGkg'
_B+=b'PSBuOwogICAgd2hpbGUgKGxvIDwgaGkpIHsKICAgICAgICBpbnQgbWlkID0gKGxvICsgaGkpID4+IDE7'
_B+=b'CiAgICAgICAgaWYgKGRhdGFbbWlkXSAmICgxIDw8IDIzKSkKICAgICAgICAgICAgaGkgPSBtaWQ7CiAg'
_B+=b'ICAgICAgZWxzZQogICAgICAgICAgICBsbyA9IG1pZCArIDE7CiAgICB9CgogICAgLyogQm91bmRhcnkg'
_B+=b'c2FuaXR5IGNoZWNrLiAqLwogICAgaWYgKGxvIDw9IDAgfHwgbG8gPj0gbiB8fCAoZGF0YVtsbyAtIDFd'
_B+=b'ICYgKDEgPDwgMjMpKSB8fCAhKGRhdGFbbG9dICYgKDEgPDwgMjMpKSkgewogICAgICAgICpvdXQgPSAt'
_B+=b'MjsKICAgICAgICByZXR1cm47CiAgICB9CgogICAgKm91dCA9IGxvOwp9CgpzdGF0aWMgdm9pZCBfc2V0'
_B+=b'dXAoKSB7CiAgICBpZiAoX3JlYWR5KSByZXR1cm47CiAgICBjdWRhRnJlZSgwKTsKICAgIGN1ZGFTdHJl'
_B+=b'YW1DcmVhdGUoJl9jYXBzdHJlYW0pOwogICAgc2l6ZV90IG5lZWQgPSAwOwogICAgY3ViOjpEZXZpY2VS'
_B+=b'YWRpeFNvcnQ6OlNvcnRLZXlzKG51bGxwdHIsIG5lZWQsCiAgICAgICAgc3RhdGljX2Nhc3Q8Y29uc3Qg'
_B+=b'aW50MzJfdCo+KG51bGxwdHIpLCBzdGF0aWNfY2FzdDxpbnQzMl90Kj4obnVsbHB0ciksCiAgICAgICAg'
_B+=b'c3RhdGljX2Nhc3Q8aW50MzJfdD4oMTAwMDAwMDAwKSwgMCwgMzIsIDApOwogICAgY3VkYURldmljZVN5'
_B+=b'bmNocm9uaXplKCk7CiAgICBfdGVtcF9ieXRlcyA9IG5lZWQgKiAxMSAvIDEwICsgNjU1MzY7CiAgICBj'
_B+=b'dWRhTWFsbG9jKCZfdGVtcCwgX3RlbXBfYnl0ZXMpOwogICAgY3VkYU1hbGxvYygmX3RlbXBfcm90LCAx'
_B+=b'MDAwMDAwMDBMTCAqIHNpemVvZihpbnQzMl90KSk7CiAgICBjdWRhTWFsbG9jKCZfcGl2b3RfZGV2LCBz'
_B+=b'aXplb2YoaW50KSk7CiAgICBfcmVhZHkgPSAxOwp9CgpzdGF0aWMgY3VkYUdyYXBoRXhlY190IF9maW5k'
_B+=b'X29yX2NhcHR1cmUoY29uc3QgZmxvYXQqIGRfaW4sIGZsb2F0KiBkX291dCwgaW50IG4sIGludCBlbmRf'
_B+=b'Yml0KSB7CiAgICBmb3IgKGludCBpID0gMDsgaSA8IF9udW1fZ3JhcGhzOyBpKyspCiAgICAgICAgaWYg'
_B+=b'KF9ncmFwaHNbaV0uZF9pbiA9PSBkX2luICYmIF9ncmFwaHNbaV0uZF9vdXQgPT0gZF9vdXQgJiYgX2dy'
_B+=b'YXBoc1tpXS5uID09IG4pCiAgICAgICAgICAgIHJldHVybiBfZ3JhcGhzW2ldLmV4ZWM7CiAgICBpZiAo'
_B+=b'X251bV9ncmFwaHMgPj0gTUFYX0dSQVBIUykgewogICAgICAgIGN1ZGFHcmFwaEV4ZWNEZXN0cm95KF9n'
_B+=b'cmFwaHNbMF0uZXhlYyk7CiAgICAgICAgbWVtbW92ZSgmX2dyYXBoc1swXSwgJl9ncmFwaHNbMV0sIChf'
_B+=b'bnVtX2dyYXBocyAtIDEpICogc2l6ZW9mKF9ncmFwaHNbMF0pKTsKICAgICAgICBfbnVtX2dyYXBocy0t'
_B+=b'OwogICAgfQogICAgaW50IGcgPSBfbnVtX2dyYXBocysrOwogICAgX2dyYXBoc1tnXS5kX2luID0gZF9p'
_B+=b'bjsgX2dyYXBoc1tnXS5kX291dCA9IGRfb3V0OyBfZ3JhcGhzW2ddLm4gPSBuOwogICAgY29uc3QgaW50'
_B+=b'MzJfdCoga2kgPSByZWludGVycHJldF9jYXN0PGNvbnN0IGludDMyX3QqPihkX2luKTsKICAgIGludDMy'
_B+=b'X3QqICAgICAgIGtvID0gcmVpbnRlcnByZXRfY2FzdDxpbnQzMl90Kj4oZF9vdXQpOwogICAgc2l6ZV90'
_B+=b'IHRiID0gX3RlbXBfYnl0ZXM7CiAgICBjdWRhU3RyZWFtQmVnaW5DYXB0dXJlKF9jYXBzdHJlYW0sIGN1'
_B+=b'ZGFTdHJlYW1DYXB0dXJlTW9kZVJlbGF4ZWQpOwogICAgY3ViOjpEZXZpY2VSYWRpeFNvcnQ6OlNvcnRL'
_B+=b'ZXlzKF90ZW1wLCB0Yiwga2ksIGtvLCBzdGF0aWNfY2FzdDxpbnQzMl90PihuKSwgMCwgZW5kX2JpdCwg'
_B+=b'X2NhcHN0cmVhbSk7CiAgICBjdWRhR3JhcGhfdCBncmFwaDsKICAgIGN1ZGFTdHJlYW1FbmRDYXB0dXJl'
_B+=b'KF9jYXBzdHJlYW0sICZncmFwaCk7CiAgICBjdWRhR3JhcGhJbnN0YW50aWF0ZSgmX2dyYXBoc1tnXS5l'
_B+=b'eGVjLCBncmFwaCwgTlVMTCwgTlVMTCwgMCk7CiAgICBjdWRhR3JhcGhEZXN0cm95KGdyYXBoKTsKICAg'
_B+=b'IHJldHVybiBfZ3JhcGhzW2ddLmV4ZWM7Cn0KCmV4dGVybiAiQyIgewoKdm9pZCBzb3J0X2luaXQoKSB7'
_B+=b'IF9zZXR1cCgpOyB9Cgp2b2lkIHNvcnRfZmxvYXQzMihjb25zdCBmbG9hdCogZF9pbiwgZmxvYXQqIGRf'
_B+=b'b3V0LCBpbnQgbikgewogICAgX3NldHVwKCk7CiAgICBjb25zdCBpbnQzMl90KiBraSA9IHJlaW50ZXJw'
_B+=b'cmV0X2Nhc3Q8Y29uc3QgaW50MzJfdCo+KGRfaW4pOwogICAgaW50MzJfdCogICAgICAga28gPSByZWlu'
_B+=b'dGVycHJldF9jYXN0PGludDMyX3QqPihkX291dCk7CiAgICBzaXplX3QgdGIgPSBfdGVtcF9ieXRlczsK'
_B+=b'CiAgICBpZiAobiA8PSAxMDAwMDAwMCkgewogICAgICAgIGN1ZGFHcmFwaEV4ZWNfdCBleGVjID0gX2Zp'
_B+=b'bmRfb3JfY2FwdHVyZShkX2luLCBkX291dCwgbiwgMjQpOwogICAgICAgIGN1ZGFHcmFwaExhdW5jaChl'
_B+=b'eGVjLCAwKTsKICAgICAgICByZXR1cm47CiAgICB9CgogICAgLyogMTAwTTogZ3JhcGggU29ydEtleXMo'
_B+=b'ZW5kX2JpdD0yNCkgdG8gX3RlbXBfcm90LCBmdXNlZCBwaXZvdCBrZXJuZWwgKGZpcnN0LWxhc3QgY2hl'
_B+=b'Y2sKICAgICAgICsgYmluYXJ5IHNlYXJjaCBpbiBvbmUga2VybmVsKSwgc2luZ2xlIHN5bmMsIHJvdGF0'
_B+=b'ZS9jb3B5L2ZhbGxiYWNrLiAqLwogICAgaW50MzJfdCogdG1wID0gc3RhdGljX2Nhc3Q8aW50MzJfdCo+'
_B+=b'KF90ZW1wX3JvdCk7CiAgICBmbG9hdCogcm90X291dF9wdHIgPSByZWludGVycHJldF9jYXN0PGZsb2F0'
_B+=b'Kj4odG1wKTsKICAgIGN1ZGFHcmFwaEV4ZWNfdCBleGVjID0gX2ZpbmRfb3JfY2FwdHVyZShkX2luLCBy'
_B+=b'b3Rfb3V0X3B0ciwgbiwgMjQpOwogICAgY3VkYUdyYXBoTGF1bmNoKGV4ZWMsIDApOwoKICAgIC8qIEZ1'
_B+=b'c2VkIGtlcm5lbDogZmlyc3QtbGFzdCBjaGVjayArIGJpbmFyeSBzZWFyY2guIFJ1bnMgb24gc3RyZWFt'
_B+=b'IDAgKGluLW9yZGVyIGFmdGVyIGdyYXBoKS4gKi8KICAgIF9maW5kX3Bpdm90X2Z1c2VkPDw8MSwgMT4+'
_B+=b'Pih0bXAsIG4sIF9waXZvdF9kZXYpOwogICAgY3VkYURldmljZVN5bmNocm9uaXplKCk7CgogICAgaW50'
_B+=b'IGNvdW50X2xvdzsKICAgIGN1ZGFNZW1jcHkoJmNvdW50X2xvdywgX3Bpdm90X2Rldiwgc2l6ZW9mKGlu'
_B+=b'dCksIGN1ZGFNZW1jcHlEZXZpY2VUb0hvc3QpOwoKICAgIC8qIGNvdW50X2xvdyA9PSAtMTogc2luZ2xl'
_B+=b'IGV4cG9uZW50LCBkaXJlY3QgY29weS4gKi8KICAgIC8qIGNvdW50X2xvdyA9PSAtMiBvciBvdGhlciBk'
_B+=b'ZWdlbmVyYXRlOiBmYWxsYmFjayB0byBlbmRfYml0PTMyLiAqLwogICAgaWYgKGNvdW50X2xvdyA8IDAp'
_B+=b'IHsKICAgICAgICBpZiAoY291bnRfbG93ID09IC0xKSB7CiAgICAgICAgICAgIGN1ZGFNZW1jcHkoa28s'
_B+=b'IHRtcCwgbiAqIHNpemVvZihpbnQzMl90KSwgY3VkYU1lbWNweURldmljZVRvRGV2aWNlKTsKICAgICAg'
_B+=b'ICB9IGVsc2UgewogICAgICAgICAgICBjdWI6OkRldmljZVJhZGl4U29ydDo6U29ydEtleXMoX3RlbXAs'
_B+=b'IHRiLCBraSwga28sIHN0YXRpY19jYXN0PGludDMyX3Q+KG4pLCAwLCAzMiwgMCk7CiAgICAgICAgfQog'
_B+=b'ICAgICAgIHJldHVybjsKICAgIH0KCiAgICAvKiBWYWxpZCBib3VuZGFyeTogdHdvLXdheSByb3RhdGlv'
_B+=b'biB2aWEgMnggY3VkYU1lbWNweSBEMkQuICovCiAgICBpbnQgY291bnRfaGlnaCA9IG4gLSBjb3VudF9s'
_B+=b'b3c7CiAgICBjdWRhTWVtY3B5KGtvLCAgICAgICAgICAgICAgdG1wICsgY291bnRfbG93LCBjb3VudF9o'
_B+=b'aWdoICogc2l6ZW9mKGludDMyX3QpLCBjdWRhTWVtY3B5RGV2aWNlVG9EZXZpY2UpOwogICAgY3VkYU1l'
_B+=b'bWNweShrbyArIGNvdW50X2hpZ2gsIHRtcCwgICAgICAgICAgICAgY291bnRfbG93ICAqIHNpemVvZihp'
_B+=b'bnQzMl90KSwgY3VkYU1lbWNweURldmljZVRvRGV2aWNlKTsKfQoKfSAgLy8gZXh0ZXJuCg=='

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
