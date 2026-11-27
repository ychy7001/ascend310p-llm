"""Build the production async server from the pinned R2T2 sources."""

import ast
from collections import Counter
from copy import deepcopy
from itertools import takewhile
from pathlib import Path


METHODS = {"init_streaming_state", "streaming_transcribe", "streaming_transcribe_no_reset",
           "finish_streaming_transcribe", "finish_streaming_transcribe_no_reset"}
EXPECTED_ROUTES = {
    "asr_stream_api_v0": Counter(init_streaming_state=5, streaming_transcribe=1, finish_streaming_transcribe=5),
    "asr_stream_api_v1": Counter(init_streaming_state=3, streaming_transcribe_no_reset=1,
                                 finish_streaming_transcribe_no_reset=2),
}


class DispatchCalls(ast.NodeTransformer):
    def __init__(self):
        self.calls = Counter()

    def visit_Assign(self, node):
        if len(node.targets) == 1 and ast.unparse(node.targets[0]) == "asr_model":
            self.calls["load"] += 1
            return ast.parse("asr_model = build_async_asr(R2T2ASRModel, args.asr_model_path)\n"
                             "asr_dispatch.engine = asr_model.model").body
        return self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and ast.unparse(node.func.value) == "asr_model":
            name = node.func.attr
            self.calls[name] += 1
            return ast.Await(value=ast.Call(func=ast.parse("asr_dispatch.call", mode="eval").body,
                                            args=[node.func, *node.args], keywords=node.keywords))
        return node


def index(body, code):
    found, = (i for i, node in enumerate(body) if ast.unparse(node) == code)
    return found


def guard_route(fn):
    body = fn.body
    header = index(body, "header_raw = await ws.recv()")
    assert index(body, "header = json.loads(header_raw)") == header + 1
    body[header:header + 2] = ast.parse("""\
header = await stream_guard.read_header(ws, secret_key_list)
if header is None:
    return
""").body
    queue = index(body, "recv_queue: asyncio.Queue = asyncio.Queue()")
    body[queue] = ast.parse("recv_queue = stream_guard.queue()").body[0]
    receiver, = (node for node in body if isinstance(node, ast.AsyncFunctionDef) and node.name == "receiver")
    body.remove(receiver)
    processor, = (node for node in body if isinstance(node, ast.AsyncFunctionDef) and node.name == "processor")
    # The guard closes the socket after EOS and processor cleanup.
    for node in ast.walk(processor):
        if isinstance(node, ast.Expr) and ast.unparse(node) == "await ws.close()":
            node.value = ast.Constant(None)
    tasks = index(body, "receiver_task = asyncio.create_task(receiver())")
    teardown = body[tasks + 1]
    close = index(teardown.finalbody, "await ws.close()")
    counted, = (i for i, node in enumerate(teardown.finalbody)
                if isinstance(node, ast.If) and ast.unparse(node.test) == "counted")
    supervisor = ast.parse("""\
reason = None
try:
    reason = await stream_guard.run(ws, recv_queue, YOUDAO_ONETIME_ASR_EOS_STRING, processor, RECV_TIMEOUT)
finally:
    await stream_guard.finish(ws, header['requestId'], reason)
""").body
    supervisor[-1].finalbody[0:0] = teardown.finalbody[counted:close]
    body[tasks:] = supervisor
    admission = ast.parse("""\
async with stream_guard.admit(ws, header['requestId']) as admitted:
    if not admitted:
        return
""").body[0]
    admission.body.extend(body[header + 2:])
    body[header + 2:] = [admission]


class GenerationYield(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_Call(self, node):
        if ast.unparse(node.func) != "self.model.generate":
            return self.generic_visit(node)
        assert len(node.args) == 1 and ast.unparse(node.args[0]) == "[inp]"
        assert [kw.arg for kw in node.keywords] == ["sampling_params", "use_tqdm"]
        assert ast.unparse(node.keywords[1].value) == "False"
        self.count += 1
        return ast.Yield(value=ast.Tuple(elts=[node.args[0].elts[0], node.keywords[0].value], ctx=ast.Load()))


def write_tree(path, tree):
    source = path.read_text(encoding="utf-8").splitlines(keepends=True)
    notice = "".join(takewhile(lambda line: line.startswith("#") or not line.strip(), source))
    result = notice + "# Modified for the ASR 310P async image.\n\n" + ast.unparse(tree) + "\n"
    compile(result, str(path), "exec")
    path.write_text(result, encoding="utf-8")


def prepare_model(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    klass, = (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "R2T2ASRModel")
    methods = [node for node in klass.body if isinstance(node, ast.FunctionDef)
               and node.name in METHODS - {"init_streaming_state"}]
    assert len(methods) == 4
    for method in methods:
        continuation = deepcopy(method)
        continuation.name = "_async_" + method.name
        rewrite = GenerationYield()
        rewrite.visit(continuation)
        assert rewrite.count == 2
        klass.body.append(continuation)
    write_tree(path, tree)


def prepare_server(path):
    source = path.read_text(encoding="utf-8").replace("use_gpu=True,", "use_gpu=False,")
    tree = ast.parse(source)
    functions = {fn.name: fn for fn in tree.body if isinstance(fn, ast.AsyncFunctionDef)}
    for name, expected in EXPECTED_ROUTES.items():
        fn = functions[name]
        calls = DispatchCalls()
        calls.visit(fn)
        assert calls.calls == expected
        guard_route(fn)
    startup = DispatchCalls()
    startup.visit(functions["initialize_models"])
    assert startup.calls == Counter(load=1, init_streaming_state=1, streaming_transcribe=1,
                                    finish_streaming_transcribe=1)

    tree.body[0:0] = ast.parse("""\
from ws_admission import StreamGuard
from ws_async_llm import AsyncLLMDispatch, build_async_asr
stream_guard = StreamGuard()
asr_dispatch = AsyncLLMDispatch()
""").body
    app = index(tree.body, "app = Sanic('onetime_asr_stream_server')")
    tree.body[app + 1:app + 1] = ast.parse("""\
app.config.WEBSOCKET_MAX_SIZE = 65536

@app.listener('after_server_stop')
async def close_dispatch(app):
    await asr_dispatch.aclose()
""").body
    secret, = (i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
               and ast.unparse(node.targets[0]) == "secret_key_list")
    tree.body[secret] = ast.parse('secret_key_list = [os.environ["R2T2_WS_SECRET"]]').body[0]

    # Importing the module must not consume the caller's command line.
    cli = tree.body.pop(index(tree.body, "args = args_parser()"))
    main = tree.body[-1]
    main.body.insert(0, cli)
    write_tree(path, tree)


if __name__ == "__main__":
    prepare_server(Path("ws_server.py"))
    prepare_model(Path("r2t2/r2t2_asr.py"))
