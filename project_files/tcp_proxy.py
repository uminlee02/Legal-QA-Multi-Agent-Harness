"""tcp_proxy.py — 듀얼스택(IPv4+IPv6) TCP 프록시. VS Code 포트포워딩 우회용.
   사용: python tcp_proxy.py <listen_port> <target_port>   (target은 127.0.0.1)
"""
import asyncio
import socket
import sys

LISTEN = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 7860


async def pipe(r, w):
    try:
        while True:
            d = await r.read(65536)
            if not d:
                break
            w.write(d)
            await w.drain()
    except Exception:
        pass
    finally:
        try:
            w.close()
        except Exception:
            pass


async def handle(cr, cw):
    try:
        sr, sw = await asyncio.open_connection("127.0.0.1", TARGET)
    except Exception:
        try:
            cw.close()
        except Exception:
            pass
        return
    await asyncio.gather(pipe(cr, sw), pipe(sr, cw))


async def main():
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("::", LISTEN))
    s.listen(128)
    s.setblocking(False)
    print(f"[proxy] [::]:{LISTEN} (IPv4+IPv6) → 127.0.0.1:{TARGET}", flush=True)
    srv = await asyncio.start_server(handle, sock=s)
    async with srv:
        await srv.serve_forever()


asyncio.run(main())
