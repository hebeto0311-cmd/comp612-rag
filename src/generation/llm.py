"""Minimal OpenAI-compatible chat client. Keys stay in memory/environment."""
import json
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.key = api_key or os.getenv('RAG_API_KEY') or os.getenv('DEEPSEEK_API_KEY')
        self.base_url = (base_url or os.getenv('RAG_BASE_URL', 'https://api.deepseek.com')).rstrip('/')
        self.model = model or os.getenv('RAG_MODEL', 'deepseek-chat')

    def complete(self, messages):
        if not self.key:
            raise LLMError('请在侧栏填写 API Key，或设置 RAG_API_KEY。')
        body = json.dumps({'model': self.model, 'messages': messages,
                           'temperature': 0, 'max_tokens': 1800}).encode()
        request = Request(self.base_url + '/chat/completions', data=body,
                          headers={'Authorization': 'Bearer ' + self.key,
                                   'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=90) as response:
                result = json.load(response)
            content = result['choices'][0]['message']['content']
            if not isinstance(content, str) or not content.strip():
                raise ValueError('empty response')
            return content
        except HTTPError as exc:
            # Never display response bodies: providers may echo sensitive input.
            raise LLMError('模型服务返回 HTTP %s，请检查 Key、余额、模型名和地址。' % exc.code) from None
        except (URLError, TimeoutError, OSError):
            raise LLMError('模型连接失败或超时，请检查网络及服务地址。') from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError('模型服务返回格式异常。') from None
