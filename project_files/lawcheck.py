"""
lawcheck.py — 법제처 인용 검증기 (이 연구의 심장 ⭐)

  extract_citations(text) : "민법 제750조 제2항 제3호" 류 인용을 정규식으로 추출
  LawVerifier.verify(c)   : 법제처 Open API로 실존/불일치 교차검증 (응답 캐시 포함)

레퍼런스: github.com/chrisryugj/korean-law-mcp (소스에서 확인해 이식, 엔드포인트 추측 X)
  - JO 포맷  : buildJO = 조4자리 + 가지2자리            (src/lib/law-parser.ts)
  - 판정 로직: 법령검색→조문조회→항확인, ✓/✗/⚠      (src/tools/verify-citations.ts)
  - 약칭맵   : 화관법→화학물질관리법 등                (src/lib/search-normalizer.ts)

법제처 API:
  검색 : GET {BASE}/lawSearch.do?OC=&type=XML&target=law&query=<법령명>&display=100
         → <law> 블록마다 <법령명한글> <법령ID> <법령일련번호>(=MST)
  조문 : GET {BASE}/lawService.do?OC=&type=JSON&target=eflaw&MST=<mst>&JO=<6자리>
         → 법령.조문.조문단위[] (조문여부="조문" 인 것), 각 항은 항번호(원숫자 ①②③ 가능)
"""
import re
import json
import time
import hashlib
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

import config

# ════════════════════════════════════════════════════════════════════════════
# 1. 약칭 → 정식명 맵  (search-normalizer.ts 의 LAW_ALIAS_ENTRIES 이식)
# ════════════════════════════════════════════════════════════════════════════
_ALIAS_ENTRIES = [
    ("대한민국헌법", ["헌법"]),
    ("상법", ["상사법"]),
    ("자유무역협정의 이행을 위한 관세법의 특례에 관한 법률",
        ["fta특례법", "에프티에이특례법"]),
    ("화학물질관리법", ["화관법", "화학물질관리법"]),
    ("행정기본법", ["행정법"]),
    ("대외무역법", ["무역법", "원산지법"]),
    ("원산지표시법", ["원산지표시"]),
    ("관세법 시행령", ["관시령", "관세시행령", "관세법시행령"]),
    ("관세법 시행규칙", ["관시규", "관세시행규칙", "관세법시행규칙"]),
    ("지방공무원법", ["지공법"]),
    ("지방공무원 임용령", ["지방공무원임용령", "지공임용령"]),
    ("지방공무원 보수규정", ["지방공무원보수규정", "지공보수규정"]),
    ("산업안전보건법", ["산안법"]),
    ("산업안전보건기준에 관한 규칙",
        ["산안기준규칙", "안전보건규칙", "산업안전보건규칙", "산안규칙", "안전보건기준규칙"]),
    ("중대재해 처벌 등에 관한 법률", ["중대재해처벌법", "중처법", "중대재해법"]),
    ("근로기준법", ["근기법", "근로법"]),
    ("남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률", ["남녀고용평등법", "고평법"]),
    ("개인정보 보호법", ["개보법", "개인정보법", "개인정보보호법"]),
    ("정보통신망 이용촉진 및 정보보호 등에 관한 법률", ["정보통신망법", "정통망법"]),
    ("부정청탁 및 금품등 수수의 금지에 관한 법률", ["청탁금지법", "김영란법"]),
    ("공직자의 이해충돌 방지법", ["이해충돌방지법", "공직자이해충돌방지법"]),
    ("국가를 당사자로 하는 계약에 관한 법률", ["국가계약법"]),
    ("지방자치단체를 당사자로 하는 계약에 관한 법률", ["지방계약법"]),
    ("공공기관의 정보공개에 관한 법률", ["정보공개법"]),
    ("부동산 거래신고 등에 관한 법률", ["부동산거래신고법", "부거법"]),
    ("주택임대차보호법", ["주임법"]),
    ("상가건물 임대차보호법", ["상임법", "상가임대차법"]),
    ("소방시설 설치 및 관리에 관한 법률", ["소방시설법"]),
    ("국세기본법", ["국기법"]),
    ("부가가치세법", ["부가세법"]),
    ("독점규제 및 공정거래에 관한 법률", ["공정거래법", "공거법", "독점규제법"]),
    ("하도급거래 공정화에 관한 법률", ["하도급법"]),
    ("약관의 규제에 관한 법률", ["약관법", "약관규제법"]),
]
# alias(공백 제거) → canonical
LAW_ALIASES = {
    a.replace(" ", ""): canon
    for canon, aliases in _ALIAS_ENTRIES
    for a in aliases
}


def resolve_alias(name: str) -> str:
    """약칭이면 정식 법령명으로, 아니면 원본 반환."""
    return LAW_ALIASES.get(name.replace(" ", ""), name)


# ════════════════════════════════════════════════════════════════════════════
# 2. 인용 추출  (verify-citations.ts 의 parseCitations 이식)
# ════════════════════════════════════════════════════════════════════════════
# "제N조", "제N조의M", "제N조 제K항 제L호"
ARTICLE_RE = re.compile(
    r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?(?:\s*제\s*(\d+)\s*항)?(?:\s*제\s*(\d+)\s*호)?"
)
# 조문 인용 직전 문맥에서 법령명 역추적: "...법률/법/시행령/시행규칙/규칙/규정/조례" 로 끝나는 것.
# 법령명 뒤에 「」『』《》 따옴표·공백이 끼어도 인식 (예: 「민법」 제750조 → "민법").
LAW_NAME_RE = re.compile(
    r"([가-힣][가-힣·ㆍ\s]{0,40}?(?:법률|법|시행령|시행규칙|규칙|규정|조례))[」』》〉\)\]\"'\s]*$"
)
# 법령명 앞 접속사·부사 제거: "또한 상법" → "상법"
STOPWORDS_RE = re.compile(
    r"^(또한|그리고|하며|따라서|따라|위해|위하여|의한|의하여|따른|해당|관련|이에|아울러|"
    r"본|이|저|그|또|및|또는|혹은|한편|더불어|이어|이는|즉|결국|결과적으로|실제로|특히)\s+"
)


@dataclass
class Citation:
    raw: str
    law_name: Optional[str]
    jo: int
    jo_branch: int            # 가지번호, 없으면 0  (제750조의2 → 2)
    hang: Optional[int]       # 항
    ho: Optional[int]         # 호
    pos: int = 0              # 본문 내 인용 시작 위치 (엄격모드 문맥 추출용)
    claimed_title: Optional[str] = None  # 모델이 단 '제N조(○○)' 주장 제목

    @property
    def jo_code(self) -> str:
        """JO 6자리 = 조(4) + 가지(2).  buildJO 이식."""
        return f"{self.jo:04d}{self.jo_branch:02d}"

    @property
    def display(self) -> str:
        s = f"제{self.jo}조" + (f"의{self.jo_branch}" if self.jo_branch else "")
        if self.hang:
            s += f" 제{self.hang}항"
        if self.ho:
            s += f" 제{self.ho}호"
        return s


def extract_citations(text: str, max_citations: int = 30) -> list[Citation]:
    out: list[Citation] = []
    seen: set = set()
    for m in ARTICLE_RE.finditer(text):
        jo = int(m.group(1))
        branch = int(m.group(2)) if m.group(2) else 0
        hang = int(m.group(3)) if m.group(3) else None
        ho = int(m.group(4)) if m.group(4) else None

        # 직전 45자에서 법령명 역추적 (긴 법령명 + 「」 여유)
        lookback = text[max(0, m.start() - 45):m.start()].rstrip()
        lm = LAW_NAME_RE.search(lookback)
        law_name: Optional[str] = None
        if lm:
            law_name = STOPWORDS_RE.sub("", re.sub(r"\s+", " ", lm.group(1)).strip()).strip()
            if len(law_name) < 2:
                law_name = None

        # 모델이 '제N조(○○)' 형태로 단 주장 제목 포착 (내용 정합성 엄격모드용)
        claimed = None
        pm = re.match(r"\s*[\(（]([^)）]{2,20})[\)）]", text[m.end():m.end() + 26])
        if pm:
            cand = pm.group(1).strip()
            if re.search(r"[가-힣]{2,}", cand) and not re.search(r"개정|신설|삭제|이하|생략", cand):
                claimed = cand

        key = (law_name or "_", jo, branch, hang, ho)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(m.group(0).strip(), law_name, jo, branch, hang, ho,
                            pos=m.start(), claimed_title=claimed))
        if len(out) >= max_citations:
            break
    return out


# ════════════════════════════════════════════════════════════════════════════
# 3. 검증 결과
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Verdict:
    status: str               # 실존(loose): "real" | "fake" | "uncertain"
    symbol: str               # ✓ | ✗ | ⚠
    official_name: Optional[str]
    note: str
    content: Optional[str] = None   # 내용정합(strict): "match"|"mismatch"|"unknown"|None
    content_note: str = ""
    article_text: str = ""          # 법제처 조문 원문(실존 시) — 검증 로직 불변, 데이터 노출만

    @property
    def is_fake(self) -> bool:
        """느슨한 가짜 = 조문 미실존."""
        return self.status == "fake"

    @property
    def strict_fake(self) -> bool:
        """엄격 가짜 = 미실존 OR (실존하나 제목·내용 불일치)."""
        return self.status == "fake" or (self.status == "real" and self.content == "mismatch")


# ─── XML/JSON 파싱 헬퍼 ──────────────────────────────────────────────────────
def _tag(xml: str, name: str) -> Optional[str]:
    m = re.search(rf"<{name}>(.*?)</{name}>", xml, re.S)
    if not m:
        return None
    v = m.group(1).strip()
    v = re.sub(r"^<!\[CDATA\[|\]\]>$", "", v).strip()  # CDATA 제거
    return v or None


def _to_list(v) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def _parse_hang(raw) -> Optional[int]:
    """항번호 → 숫자. 법제처는 원숫자(①②③)로 주는 경우가 많음 (parseHangNumber 이식)."""
    s = str(raw or "").strip()
    if not s:
        return None
    idx = _CIRCLED.find(s[0])
    if idx >= 0:
        return idx + 1
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None


def _norm(s: str) -> str:
    return (s or "").replace(" ", "")


# ─── 내용 정합성(엄격모드) 헬퍼 ───────────────────────────────────────────────
def _bigrams(s: str) -> set:
    s = re.sub(r"[^가-힣]", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _bigram_sim(a: str, b: str) -> float:
    """한글 글자 바이그램 자카드 유사도 (0~1). 짧은 조문제목 비교용."""
    A, B = _bigrams(a), _bigrams(b)
    return len(A & B) / len(A | B) if (A and B) else 0.0


def _hangul_tokens(s: str, min_len: int = 2) -> set:
    return {t for t in re.findall(r"[가-힣]+", s or "") if len(t) >= min_len}


def _flatten_text(v) -> str:
    if isinstance(v, str):
        return "" if v.startswith("<img") else v
    if isinstance(v, list):
        return " ".join(_flatten_text(x) for x in v)
    return ""


def _unit_text(unit: dict) -> str:
    """조문단위 객체에서 제목+본문+항/호 텍스트를 평탄화 (법제처 본문)."""
    parts = [unit.get("조문제목") or "", _flatten_text(unit.get("조문내용"))]
    for h in _to_list(unit.get("항")):
        if not isinstance(h, dict):
            continue
        parts.append(_flatten_text(h.get("항내용")))
        for ho in _to_list(h.get("호")):
            if isinstance(ho, dict):
                parts.append(_flatten_text(ho.get("호내용")))
    text = " ".join(p for p in parts if p)
    return re.sub(r"<[^>]+>", " ", text)


def _article_fulltext(unit: dict) -> str:
    """법제처 조문 원문(전문) — 조문내용 + 항/호, 줄바꿈 유지. docx '조문 원문'용."""
    parts = [_flatten_text(unit.get("조문내용"))]
    for h in _to_list(unit.get("항")):
        if not isinstance(h, dict):
            continue
        parts.append(_flatten_text(h.get("항내용")))
        for ho in _to_list(h.get("호")):
            if isinstance(ho, dict):
                parts.append(_flatten_text(ho.get("호내용")))
    text = "\n".join(p.strip() for p in parts if p and p.strip())
    return re.sub(r"<[^>]+>", " ", text).strip()


# ════════════════════════════════════════════════════════════════════════════
# 4. 검증기  (verify-citations.ts 의 verifyOne 이식)
# ════════════════════════════════════════════════════════════════════════════
class LawVerifier:
    def __init__(self, oc: Optional[str] = None, cache_dir=None):
        self.oc = oc or config.OC
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
        self._last_live = 0.0
        self.live_calls = 0
        self.cache_hits = 0

    # ── HTTP (캐시 + 레이트리밋 + 재시도) ────────────────────────────────────
    def _get(self, endpoint: str, params: dict) -> str:
        cache_params = {k: v for k, v in params.items() if k != "OC"}
        ckey = endpoint + "?" + urllib.parse.urlencode(sorted(cache_params.items()))
        cfile = self.cache_dir / (hashlib.sha1(ckey.encode("utf-8")).hexdigest() + ".txt")
        if cfile.exists():
            self.cache_hits += 1
            return cfile.read_text(encoding="utf-8")

        url = f"{config.LAW_API_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
        last_err = None
        for attempt in range(config.MAX_RETRY):
            dt = time.time() - self._last_live
            if dt < config.RATE_LIMIT_SLEEP:
                time.sleep(config.RATE_LIMIT_SLEEP - dt)
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "legal-faithfulness/1.0"})
                with urllib.request.urlopen(req, timeout=config.REQUEST_TIMEOUT) as r:
                    body = r.read().decode("utf-8", "replace")
                self._last_live = time.time()
                self.live_calls += 1
                cfile.write_text(body, encoding="utf-8")  # 에러응답도 캐싱(반복 호출 방지)
                return body
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 429:                      # 레이트리밋 → 백오프
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"법제처 API 호출 실패({endpoint}): {last_err}")

    # ── 1단계: 법령 검색 ─────────────────────────────────────────────────────
    def _search_raw(self, query: str) -> list[dict]:
        # type=JSON 사용: 이 OC는 XML 타입이 "미신청"으로 HTML 에러를 반환하지만
        # JSON은 정상. (repo도 LAW_RESPONSE_TYPE=JSON 우회를 안내)
        body = self._get("lawSearch.do",
                         {"OC": self.oc, "type": "JSON", "target": "law",
                          "query": query, "display": "100"})
        low = body.lstrip()[:200].lower()
        if "<html" in low or "<!doctype" in low:   # 미신청/에러 페이지
            return []
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return []
        laws = []
        for item in _to_list((data.get("LawSearch") or {}).get("law")):
            if not isinstance(item, dict):
                continue
            nm = item.get("법령명한글")
            if not nm:
                continue
            laws.append({"name": nm, "id": item.get("법령ID"),
                         "mst": item.get("법령일련번호")})
        return laws

    def search_law(self, name: str) -> list[dict]:
        """앞 문맥 노이즈에 강건한 법령 검색.

        인용 추출 시 법령명 앞에 잡문이 섞일 수 있음
        ('사안은 민법', '형사상으로는 형법', '민사상으로는 언론중재…법률').
        앞 토큰을 하나씩 떼어가며 검색해, '정확 일치'가 나오는 가장 긴 후보를
        우선 채택한다(멀티워드 법령명 보존). 정확 일치가 없으면 기존 동작
        (원본 → 트레일링 토큰)으로 폴백 — 가짜 법령 탐지는 그대로 유지.
        """
        resolved = resolve_alias(name)
        toks = resolved.split()
        # 후보: 앞 토큰 제거 변형(긴 것 우선) + 트레일링 법령명 토큰('…형법'→'형법')
        candidates = [" ".join(toks[i:]) for i in range(min(len(toks), 5))]
        mt = re.search(r"[가-힣]+(?:법률|법|시행령|시행규칙|규칙|규정|령|조례)\s*$", resolved)
        if mt:
            candidates.append(mt.group(0).strip())
        for cand in candidates:
            q = resolve_alias(cand)
            if len(q.replace(" ", "")) < 2:
                continue
            laws = self._search_raw(q)
            exact = [l for l in laws if _norm(l["name"]) == _norm(q)]
            if exact:
                return exact                         # 정확 일치 법령만 반환(난민법 등 잡음 제거)

        # 폴백: 원본 → 트레일링 토큰 (기존 동작, 가짜 탐지 유지)
        laws = self._search_raw(resolved)
        if laws:
            return laws
        if mt:
            q2 = resolve_alias(mt.group(0).strip())
            if q2 != resolved:
                return self._search_raw(q2)
        return []

    @staticmethod
    def _score(law_name: str, query: str) -> int:
        """관련도 점수 (scoreLawRelevance 축약)."""
        score = 0
        if _norm(law_name) == _norm(query):
            score += 100
        if _norm(law_name).startswith(_norm(query)):
            score += 80
        if _norm(query) in _norm(law_name):
            score += 40
        if not re.search(r"시행령|시행규칙", law_name):  # 본법 우선
            score += 5
        return score

    @staticmethod
    def _loose_match(target: str, official: str) -> bool:
        t, o = _norm(target), _norm(official)
        return (o == t or o.startswith(t) or t.endswith(o)   # 앞 노이즈: '형사상으로는형법'⊃'형법'
                or t.startswith(re.sub(r"(법률|법)$", "법", o)))

    # ── 2단계: 조문 조회 ─────────────────────────────────────────────────────
    def get_article_unit(self, mst: str, jo_code: str) -> Optional[dict]:
        body = self._get("lawService.do",
                         {"OC": self.oc, "type": "JSON", "target": "eflaw",
                          "MST": str(mst), "JO": jo_code})
        if "<html" in body.lower():        # 에러 페이지 = 조문 없음
            return None
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return None
        units = _to_list((((data or {}).get("법령") or {}).get("조문") or {}).get("조문단위"))
        for u in units:
            if isinstance(u, dict) and u.get("조문여부") == "조문":
                return u
        return None

    # ── 내용 정합성(엄격모드) 판정 ──────────────────────────────────────────
    # 조문이 실존할 때, 모델이 말한 내용이 그 조문과 실제로 맞는지 법제처 본문과 대조.
    #   match    : 모델 주장제목이 실제 조문제목과 유사 OR 인용 주변 문맥이 본문과 겹침
    #   mismatch : 모델이 제목을 명시했으나 실제 조문제목·본문 어느 쪽과도 안 겹침(내용 환각)
    #   unknown  : 제목 미표기 + 본문 매칭 없음 → 판정 보류
    _TITLE_SIM_TH = 0.34

    def _content_verdict(self, unit: dict, c: Citation, answer: Optional[str],
                         gold_text: str = ""):
        # 참조 본문 = 법제처 조문 본문 (+ 해당 조문이 KoBLEX gold에 있으면 gold 본문도)
        real_title = (unit.get("조문제목") or "").strip()
        body = _unit_text(unit)
        reference = real_title + " " + body + " " + (gold_text or "")
        title_sim = (_bigram_sim(c.claimed_title, real_title)
                     if (c.claimed_title and real_title) else None)
        ctx = answer[max(0, c.pos - 90): c.pos + 90] if answer else ""
        shared = _hangul_tokens(ctx) & _hangul_tokens(reference)

        if title_sim is not None and title_sim >= self._TITLE_SIM_TH:
            return "match", f"제목일치 «{c.claimed_title}»~«{real_title}»"
        if len(shared) >= 2:
            return "match", f"본문키워드 일치 {sorted(shared)[:4]}"
        if c.claimed_title and real_title:
            return "mismatch", f"내용환각: 모델 «{c.claimed_title}» ↔ 실제 «{real_title}»"
        return "unknown", "내용 대조 불가(제목 미표기·본문 매칭 없음)"

    # ── 통합 판정 ────────────────────────────────────────────────────────────
    def verify(self, c: Citation, answer: Optional[str] = None,
               strict: bool = False, gold_by_article: Optional[dict] = None) -> Verdict:
        """gold_by_article: {(법령명norm, jo, branch): gold조문본문} — KoBLEX gold 대조용(선택)."""
        if not config.oc_is_set() and self.oc in ("", "______", None):
            raise RuntimeError(
                "OC 키 미설정. open.law.go.kr 에서 발급 후 `export LAW_OC=...` 또는 config.py 수정.")

        if not c.law_name:
            return Verdict("uncertain", "⚠", None, "법령명 추출 실패 (앞 문맥에 법령명 명시 필요)")

        laws = self.search_law(c.law_name)
        if not laws:
            return Verdict("fake", "✗", None, "법제처 DB에 해당 법령 없음 (오탈자/미존재)")

        chosen = max(laws, key=lambda l: self._score(l["name"], c.law_name))
        official = chosen["name"]
        if not self._loose_match(c.law_name, official):
            return Verdict("uncertain", "⚠", official,
                           f"검색은 '{official}'(으)로만 매칭 — 법령명 재확인 필요")
        if not chosen.get("mst"):
            return Verdict("uncertain", "⚠", official, "MST 추출 실패")

        unit = self.get_article_unit(chosen["mst"], c.jo_code)
        if unit is None:
            return Verdict("fake", "✗", official, f"{c.display} — 해당 조문 없음")

        title = unit.get("조문제목") or ""
        ttl = f" ({title})" if title else ""

        def _real(note: str) -> Verdict:
            v = Verdict("real", "✓", official, note)
            v.article_text = _article_fulltext(unit)   # 법제처 원문 노출(검증 결과 불변)
            if strict:
                gold_text = ""
                if gold_by_article:
                    gold_text = (gold_by_article.get((_norm(official), c.jo, c.jo_branch))
                                 or gold_by_article.get((_norm(c.law_name or ""), c.jo, c.jo_branch))
                                 or "")
                v.content, v.content_note = self._content_verdict(unit, c, answer, gold_text)
            return v

        if c.hang:
            hnums = [n for n in (_parse_hang(h.get("항번호"))
                                 for h in _to_list(unit.get("항")) if isinstance(h, dict))
                     if n]
            if c.hang in hnums:
                return _real(f"{c.display} 실존{ttl}")
            if not hnums:
                return Verdict("uncertain", "⚠", official,
                               f"제{c.jo}조 실존하나 항 확인불가(응답형식)")
            return Verdict("fake", "✗", official,
                           f"제{c.hang}항 없음 (최대 제{max(hnums)}항)")

        return _real(f"{c.display} 실존{ttl}")
