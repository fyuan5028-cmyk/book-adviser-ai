import io
import hmac
import json
import math
import os
import re
import zipfile
from html.parser import HTMLParser

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


load_dotenv()


def get_setting(name, default=None):
    """本地从 .env 读取，部署后从 Streamlit 的加密设置读取。"""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


API_KEY = get_setting("OPENAI_API_KEY")
MODEL_NAME = get_setting("OPENAI_MODEL", "gpt-5.6-luna")
EMBEDDING_MODEL = get_setting("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
ACCESS_CODE = get_setting("APP_ACCESS_CODE", "")
MAX_AI_CALLS = int(get_setting("MAX_AI_CALLS_PER_SESSION", "12"))
KNOWLEDGE_VERSION = 3
BUILTIN_LIBRARY_FILE = os.path.join(os.path.dirname(__file__), "knowledge_cards.json")

st.set_page_config(page_title="书籍顾问 AI", page_icon="📚", layout="wide")


def load_builtin_cards():
    """读取可公开发布的提炼型知识卡片，不保存书籍原文。"""
    if not os.path.exists(BUILTIN_LIBRARY_FILE):
        return [], []
    with open(BUILTIN_LIBRARY_FILE, "r", encoding="utf-8") as file:
        cards = json.load(file)
    chunks = []
    for card in cards:
        chunks.append({
            "source": card["book"],
            "location": f"知识卡片：{card['title']}",
            "text": (
                f"主题：{card['title']}。核心原则：{card['principle']} "
                f"适用场景：{card['use_when']} 行动方法：{card['action']} "
                f"边界提醒：{card['boundary']}"
            ),
        })
    names = list(dict.fromkeys(card["book"] for card in cards))
    return chunks, names


BUILTIN_CHUNKS, BUILTIN_BOOK_NAMES = load_builtin_cards()


def show_login():
    """分享网页时，用邀请码阻止陌生人消耗额度。"""
    st.title("📚 我的书籍顾问 AI")
    st.write("输入邀请码后即可使用。你不需要填写 API 密钥。")
    entered_code = st.text_input("邀请码", type="password")
    if st.button("进入书籍顾问", type="primary"):
        entered_bytes = entered_code.encode("utf-8")
        expected_bytes = str(ACCESS_CODE).encode("utf-8")
        if hmac.compare_digest(entered_bytes, expected_bytes):
            st.session_state.authorized = True
            st.rerun()
        else:
            st.error("邀请码不正确，请向网页提供者索取。")


if "ai_calls" not in st.session_state:
    st.session_state.ai_calls = 0

if ACCESS_CODE and not st.session_state.get("authorized", False):
    show_login()
    st.stop()


def split_text(text, size=1200, overlap=150):
    """把长文字切成带少量重叠的小段。"""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    pieces = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        pieces.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return pieces


def read_pdf(uploaded_file):
    """读取文字版 PDF，并保留页码。"""
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for part_number, part in enumerate(split_text(page_text), start=1):
            chunks.append(
                {
                    "source": uploaded_file.name,
                    "location": f"第 {page_number} 页，第 {part_number} 段",
                    "text": part,
                }
            )
    return chunks


def read_txt(uploaded_file):
    """兼容常见的中文 TXT 编码。"""
    raw = uploaded_file.getvalue()
    text = None

    for encoding in ("utf-8", "gb18030", "utf-16"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        text = raw.decode("utf-8", errors="ignore")

    return [
        {
            "source": uploaded_file.name,
            "location": f"第 {number} 段",
            "text": part,
        }
        for number, part in enumerate(split_text(text), start=1)
    ]


class EpubTextExtractor(HTMLParser):
    """从 EPUB 内部的网页章节中提取可读文字。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.ignored_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.ignored_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.ignored_depth > 0:
            self.ignored_depth -= 1

    def handle_data(self, data):
        if self.ignored_depth == 0 and data.strip():
            self.parts.append(data.strip())

    def get_text(self):
        return " ".join(self.parts)


def read_epub(uploaded_file):
    """读取 EPUB 中的 HTML/XHTML 章节。"""
    chunks = []

    with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as book:
        chapter_files = sorted(
            name
            for name in book.namelist()
            if name.lower().endswith((".xhtml", ".html", ".htm"))
            and name.rsplit("/", 1)[-1].lower()
            not in {"nav.xhtml", "toc.xhtml", "toc.html", "toc.htm"}
        )

        for chapter_number, chapter_name in enumerate(chapter_files, start=1):
            raw = book.read(chapter_name)
            try:
                chapter_html = raw.decode("utf-8")
            except UnicodeDecodeError:
                chapter_html = raw.decode("utf-8", errors="ignore")

            extractor = EpubTextExtractor()
            extractor.feed(chapter_html)
            chapter_text = extractor.get_text()

            for part_number, part in enumerate(split_text(chapter_text), start=1):
                chunks.append(
                    {
                        "source": uploaded_file.name,
                        "location": f"第 {chapter_number} 章，第 {part_number} 段",
                        "text": part,
                    }
                )

    return chunks


def add_embeddings(chunks, batch_size=64):
    """把书籍片段转换成用于语义搜索的数字向量。"""
    client = OpenAI(api_key=API_KEY)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        if st.session_state.ai_calls >= MAX_AI_CALLS:
            raise RuntimeError("本次使用次数已达到上限，请稍后重新打开网页。")
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[chunk["text"] for chunk in batch],
        )
        st.session_state.ai_calls += 1
        for chunk, result in zip(batch, response.data):
            chunk["embedding"] = result.embedding

    return chunks


def cosine_similarity(first, second):
    dot_product = sum(a * b for a, b in zip(first, second))
    first_length = math.sqrt(sum(a * a for a in first))
    second_length = math.sqrt(sum(b * b for b in second))
    if first_length == 0 or second_length == 0:
        return 0.0
    return dot_product / (first_length * second_length)


def find_relevant_chunks(question, chunks, limit=8):
    """按含义而不是表面关键词搜索书籍。"""
    client = OpenAI(api_key=API_KEY)
    if st.session_state.ai_calls >= MAX_AI_CALLS:
        raise RuntimeError("本次使用次数已达到上限，请稍后重新打开网页。")
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )
    st.session_state.ai_calls += 1
    question_embedding = response.data[0].embedding

    ranked = []
    for chunk in chunks:
        score = cosine_similarity(question_embedding, chunk["embedding"])
        ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    relevant = []
    for score, chunk in ranked[:limit]:
        selected = dict(chunk)
        selected["similarity"] = score
        relevant.append(selected)
    return relevant


def format_context(chunks):
    sections = []
    for number, chunk in enumerate(chunks, start=1):
        sections.append(
            f"【资料 {number}｜{chunk['source']}｜{chunk['location']}｜"
            f"检索相关度 {chunk.get('similarity', 0):.2f}】\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(sections)


def ask_ai(instructions, user_input):
    if st.session_state.ai_calls >= MAX_AI_CALLS:
        raise RuntimeError("本次使用次数已达到上限，请稍后重新打开网页。")
    client = OpenAI(api_key=API_KEY)
    response = client.responses.create(
        model=MODEL_NAME,
        instructions=instructions,
        input=user_input,
    )
    st.session_state.ai_calls += 1
    return response.output_text


if st.session_state.get("knowledge_version") != KNOWLEDGE_VERSION:
    st.session_state.knowledge_version = KNOWLEDGE_VERSION
    st.session_state.book_chunks = []
    st.session_state.book_names = []
    st.session_state.clarifying_questions = ""
    st.session_state.final_answer = ""


if BUILTIN_CHUNKS and not st.session_state.book_chunks and API_KEY:
    try:
        with st.spinner("正在加载内置书籍知识库……"):
            st.session_state.book_chunks = add_embeddings(
                [dict(chunk) for chunk in BUILTIN_CHUNKS]
            )
            st.session_state.book_names = list(BUILTIN_BOOK_NAMES)
    except Exception as error:
        st.error("内置知识库加载失败：")
        st.code(str(error))


st.title("📚 我的书籍顾问 AI")
st.caption("提出问题 → 顾问追问 → 获得结合精选书籍知识的行动方案")

with st.sidebar:
    st.header("顾问知识库")
    if BUILTIN_BOOK_NAMES:
        st.success(f"已准备 {len(BUILTIN_BOOK_NAMES)} 本精选书籍")
        for book_name in BUILTIN_BOOK_NAMES:
            st.write(f"• {book_name}")
    else:
        st.warning("暂未找到内置知识卡片文件。")
    st.caption("书籍由网站管理者统一整理和维护，普通用户只需提出问题。")
    st.divider()
    st.caption(f"本次已使用 {st.session_state.ai_calls}/{MAX_AI_CALLS} 次 AI 请求")


if not API_KEY:
    st.error("没有读取到 API 密钥，请检查 .env 文件。")

if not st.session_state.book_chunks:
    st.info("内置知识库正在准备或尚未加载，请稍后刷新网页。")
else:
    st.success(f"知识库已准备好：{len(st.session_state.book_names)} 本书")

    st.header("第一步：描述你遇到的问题")
    question = st.text_area(
        "发生了什么？",
        height=150,
        placeholder="请尽量写清楚人物关系、发生的事情和你希望达到的结果。",
    )

    if st.button("先分析并向我追问"):
        if not question.strip():
            st.warning("请先描述你的问题。")
        else:
            relevant_chunks = find_relevant_chunks(
                question, st.session_state.book_chunks
            )
            context = format_context(relevant_chunks)

            prompt = f"""
用户遇到的问题：
{question}

从书籍中找到的相关资料：
{context}
"""

            instructions = """
你是一名理性、温和、懂得判断关系性质的人际关系书籍顾问。
现在不要急着给最终方案。先判断当前关系主要属于合作、普通分歧、谈判竞争，
还是具有明显恶意的对抗；这个判断只能作为暂定假设，并要说明依据和不确定性。

竞争和战略思想可以用于真实存在利益冲突、资源竞争、谈判或恶意对抗的情境，
但必须转化为合法、克制、可执行的人际行动。合作关系中也可以借鉴战略思维，
但不要未经证据就把对方当作敌人。

然后：
1. 用自然语言简要复述你对情况的理解；
2. 区分事实、用户的解释和仍未知的信息；
3. 判断书籍资料与问题是高度适用、部分适用还是不适用，并说明原因；
4. 提出 3 到 5 个真正会改变建议的追问。

检索相关度只表示文字含义可能接近，不代表观点一定适用。
不要编造书中不存在的观点。引用资料时注明资料编号。
"""

            try:
                with st.spinner("顾问正在阅读并分析……"):
                    st.session_state.clarifying_questions = ask_ai(
                        instructions, prompt
                    )
                st.session_state.final_answer = ""
            except Exception as error:
                st.error("调用 AI 时出现错误：")
                st.code(str(error))

    if st.session_state.clarifying_questions:
        st.subheader("顾问的初步分析与追问")
        st.write(st.session_state.clarifying_questions)

        st.header("第二步：补充信息")
        extra_information = st.text_area(
            "回答上面的追问，或者补充你认为重要的情况",
            height=180,
        )

        if st.button("根据书籍给出完整方案", type="primary"):
            combined_question = f"{question}\n{extra_information}"
            relevant_chunks = find_relevant_chunks(
                combined_question, st.session_state.book_chunks
            )
            context = format_context(relevant_chunks)

            prompt = f"""
用户最初的问题：
{question}

顾问上一轮的分析与追问：
{st.session_state.clarifying_questions}

用户补充的信息：
{extra_information if extra_information.strip() else "用户暂未补充更多信息。"}

从书籍中找到的相关资料：
{context}
"""

            instructions = """
你是一名理性、尊重边界、能够综合而非照搬书本的人际关系书籍顾问。
请依据用户信息进行独立判断，再决定书籍观点是否适用，不要把猜测说成事实。

先判断关系性质：合作、普通分歧、谈判竞争或明显恶意对抗。
当确有利益冲突、资源竞争、谈判或恶意对抗时，可以采用竞争和战略思想；
将其转化为信息判断、时机选择、成本控制、边界、谈判与风险防范等现实行动。
在合作关系中也允许借鉴战略思维，但不能因为书中使用战争语言就默认对方是敌人。
不要建议违法、欺骗、报复、羞辱、威胁或伤害行为。

检索相关度仅用于找资料，不代表资料天然正确或适用。
请使用清晰的中文，并按以下结构输出：

## 核心判断
说明关系性质、关键矛盾，以及仍然不确定之处。

## 书籍适用性
明确判断书籍是高度适用、部分适用还是不适用，并解释原因。

## 可借鉴的书中原则
只列出确实适用的原则，用“资料 1”“资料 2”标明来源；
解释如何转换到当前情境，而不是复述或生硬套用原文。
若没有合适原则，直接说没有，并把后续内容标为“通用建议”。

## 行动方案
综合现实信息和适用的书籍原则，给出今天、接下来三天、以后可以采取的行动。

## 可以直接说的话
提供一段自然、不施压、不操控他人的沟通示例。

## 风险和边界
说明哪些行为应当避免，以及什么情况下应寻求可信任的人或专业帮助。

如果书籍资料不足以支持某个判断，请明确说“书籍资料不足”，
然后可以给出清楚标注的通用人际建议，不要编造书中内容。
"""

            try:
                with st.spinner("顾问正在整理完整方案……"):
                    st.session_state.final_answer = ask_ai(instructions, prompt)
            except Exception as error:
                st.error("调用 AI 时出现错误：")
                st.code(str(error))

    if st.session_state.final_answer:
        st.subheader("完整建议")
        st.write(st.session_state.final_answer)
        st.caption("AI 建议仅供参考。涉及安全、心理健康、法律或医疗问题时，请咨询专业人士。")
