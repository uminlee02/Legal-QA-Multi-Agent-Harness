"""launch_dual.py — IPv4+IPv6 듀얼스택 소켓으로 uvicorn 실행.

VS Code 원격 포트 포워딩은 localhost를 IPv6(::1)로 풀고, SSH/직접접속은 IPv4(127.0.0.1)를
쓰기 때문에 양쪽 다 리슨해야 한다. uvicorn --host 는 한 패밀리만 잡으므로, 여기서 듀얼스택
소켓(IPV6_V6ONLY=0)을 만들어 uvicorn --fd 로 넘긴다.
"""
import os
import socket

PORT = int(os.environ.get("PORT", "7860"))

s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)   # IPv4도 수용(듀얼스택)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("::", PORT))
s.listen(128)
os.set_inheritable(s.fileno(), True)
print(f"[dual] listening on [::]:{PORT} (IPv4+IPv6), fd={s.fileno()}", flush=True)
os.execv("/opt/conda/bin/uvicorn", ["uvicorn", "app:app", "--fd", str(s.fileno())])
