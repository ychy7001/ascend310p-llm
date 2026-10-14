"""候选服务普通文本、JSON Schema 与并发变长生成回归。"""

import argparse
import concurrent.futures
import json
import time
import urllib.request


def request(url, index, structured):
    expected = str(40 + index)
    body = {
        "model": "qwen36-w4a8-c8",
        "temperature": 0,
        "max_tokens": 64,
        "seed": 17856,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": ("请按要求回答。" * (index * 7))
                + f"计算 {20 + index}+20。只输出答案数字，不要解释。",
            }
        ],
    }
    if structured:
        body["messages"][0]["content"] += (
            "用 JSON 对象回答，字段 answer 为答案的字符串。"
        )
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        }
    start = time.monotonic()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.load(response)
        content = data["choices"][0]["message"]["content"]
        answer = json.loads(content)["answer"] if structured else content.strip()
        return {
            "status": "PASS" if answer == expected else "FAIL",
            "index": index,
            "structured": structured,
            "content": content,
            "expected": expected,
            "seconds": time.monotonic() - start,
            "usage": data.get("usage"),
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {
            "status": "ERROR",
            "index": index,
            "structured": structured,
            "error": str(error),
            "seconds": time.monotonic() - start,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18029")
    args = parser.parse_args()
    records = []
    for structured in [False, True]:
        for repeat in range(2):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(request, args.url, i, structured) for i in range(4)
                ]
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    record["repeat"] = repeat
                    records.append(record)
                    print(json.dumps(record, ensure_ascii=False), flush=True)
    assert all(record["status"] == "PASS" for record in records), (
        "generation regression failed"
    )
    print(
        json.dumps(
            {
                "passed": len(records),
                "scope": "short generation smoke, not full accuracy evaluation",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
