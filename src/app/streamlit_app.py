"""Run: python -m streamlit run src/app/streamlit_app.py"""
from pathlib import Path
import os
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import streamlit as st
from generation.answer import AnswerService
from generation.llm import LLM, LLMError

st.set_page_config(page_title='Python 文档问答', page_icon='🐍', layout='wide', initial_sidebar_state='collapsed')
st.markdown('<style>' + (Path(__file__).with_name('style.css')).read_text(encoding='utf-8') + '</style>', unsafe_allow_html=True)
title_col, settings_col = st.columns([5, 1], vertical_alignment='center')
with title_col:
    st.markdown('<div class="page-heading"><h1>Python 文档问答</h1></div>', unsafe_allow_html=True)
with settings_col:
    with st.popover('设置', use_container_width=False):
        st.header('模型连接')
        base_url = st.text_input('API 服务地址', value=os.getenv('RAG_BASE_URL', 'https://api.deepseek.com'))
        model = st.text_input('模型名称', value=os.getenv('RAG_MODEL', 'deepseek-chat'))
        key = st.text_input('API Key', type='password', help='仅用于本次会话的模型请求，不写入项目文件。')
        st.divider()
        st.header('检索设置')
        mode = st.selectbox('检索模式', ['hybrid', 'sparse', 'dense'],
                            format_func=lambda x: {'hybrid': '混合检索', 'sparse': '关键词检索', 'dense': '向量检索'}[x])
        variant = st.selectbox('数据规模', ['large', 'medium', 'small'])
        k = st.slider('参考片段数', 1, 10, 5)
        use_cache = st.checkbox('使用答案缓存', value=True)

with st.container(border=True, key='composer'):
    query = st.chat_input('例如：如何反转一个列表？', max_chars=2000, key='question')

submit = bool(query and query.strip())
if submit:
    service = AnswerService(llm=LLM(api_key=key or None, base_url=base_url, model=model))
    try:
        with st.spinner('正在回答…'):
            result = service.answer(query, mode=mode, variant=variant, k=k, use_cache=use_cache)
        st.session_state['last_answer'] = (query, result, service.last_hits, service.cache_hit, mode, variant)
    except LLMError as exc:
        st.error(str(exc))
        st.session_state.pop('last_answer', None)
    except ImportError:
        st.error('缺少检索依赖，请按 README_C 安装依赖后重启。')
        st.session_state.pop('last_answer', None)
    except Exception:
        st.error('检索或生成失败。请检查索引文件是否齐全、BGE 模型能否下载，以及网络连接。')
        st.session_state.pop('last_answer', None)

if 'last_answer' in st.session_state:
    question, result, hits, cached, used_mode, used_variant = st.session_state['last_answer']
    st.subheader('回答')
    st.caption('问题：' + question)
    if result['refused']:
        st.info(result['answer'])
    else:
        with st.container(border=True, key='answer'):
            st.markdown(result['answer'])
    if result['citations']:
        st.subheader('参考来源')
    by_id = {h['chunk_id']: h for h in hits}
    for number, cid in enumerate(result['citations'], 1):
        h = by_id.get(cid)
        if not h:
            continue
        with st.expander(str(number).zfill(2) + '  /  ' + h.get('section_path', 'Python 文档'), expanded=False):
            url = h.get('url', '')
            if urlparse(url).scheme == 'https' and urlparse(url).hostname == 'docs.python.org':
                st.link_button('打开 Python 官方原文', url)
            with st.container(height=220):
                st.write(h['text'])
    with st.expander('查看本次检索片段'):
        for h in hits:
            st.write(h['chunk_id'], '分数：', h['score'])
            st.text(h['text'])

