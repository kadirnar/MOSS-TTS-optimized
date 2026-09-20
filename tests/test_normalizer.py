import pytest

from moss_tts._model.tts_robust_normalizer_single_script import normalize_tts_text

TEST_CASES = [
    # 1) .map / dot-leading token / filenames / version numbers
    (
        "dot_map_sentence",
        "2026 年 3 月 31 日，安全研究员 Chaofan Shou (@Fried_rice) 发现 Anthropic 的 npm 包中暴露了 .map 文件，",
        "2026年3月31日，安全研究员 Chaofan Shou (@Fried_rice) 发现 Anthropic 的 npm 包中暴露了 .map 文件，",
    ),
    (
        "dot_tokens",
        "别把 .env、.npmrc、.gitignore 提交上去。",
        "别把 .env、.npmrc、.gitignore 提交上去。",
    ),
    (
        "file_names",
        "请检查 bundle.min.js、package.json 和 processing_moss_tts.py。",
        "请检查 bundle.min.js、package.json 和 processing_moss_tts.py。",
    ),
    ("index_d_ts", "index.d.ts 里也有同样的问题。", "index.d.ts 里也有同样的问题。"),
    (
        "version_build",
        "Bug 的讨论可以精确到 v2.3.1 (Build 15)。",
        "Bug 的讨论可以精确到 v2.3.1 (Build 15)。",
    ),
    ("version_rc", "3.0.0-rc.1 还不能上生产。", "3.0.0-rc.1 还不能上生产。"),
    (
        "jar_name",
        "fabric-api-0.91.3+1.20.2.jar 需要单独下载。",
        "fabric-api-0.91.3+1.20.2.jar 需要单独下载。",
    ),
    # 2) URL / Email / mention / hashtag / Reddit
    (
        "url",
        "仓库地址是 https://github.com/instructkr/claude-code",
        "仓库地址是 https://github.com/instructkr/claude-code",
    ),
    ("email", "联系邮箱：ops+tts@example.ai", "联系邮箱：ops+tts@example.ai"),
    ("mention", "@Fried_rice 说这是 source map 暴露。", "@Fried_rice 说这是 source map 暴露。"),
    ("reddit", "去 r/singularity 看讨论。", "去 r/singularity 看讨论。"),
    ("hashtag_chain", "#张雪峰#张雪峰[话题]#张雪峰事件", "#张雪峰#张雪峰[话题]#张雪峰事件"),
    (
        "mention_hashtag_boundary",
        "关注@biscuit0228_并转发#thetime_tbs",
        "关注 @biscuit0228_ 并转发 #thetime_tbs",
    ),
    # 3) Brackets / control tokens: preserve their contents.
    ("speaker_bracket", "[S1]你好。[S2]收到。", "[S1]你好。[S2]收到。"),
    ("event_bracket", "请模仿 {whisper} 的语气说“别出声”。", "请模仿 {whisper} 的语气说“别出声”。"),
    ("order_bracket", "订单号：[AB-1234-XYZ]", "订单号：[AB-1234-XYZ]"),
    # 4) Structural symbols: replace with boundaries.
    ("struct_headline", "〖重磅〗《新品发布》——现在开始！", "重磅。新品发布。现在开始！"),
    (
        "struct_notice",
        "【公告】今天 20:00 维护——预计 30 分钟。",
        "公告。今天20:00维护。预计30分钟。",
    ),
    ("struct_quote_chain", "『特别提醒』「不要外传」", "特别提醒。不要外传。"),
    (
        "flow_arrow_chain",
        "请求接入 -> 身份与策略判定 -> 域服务处理",
        "请求接入，身份与策略判定，域服务处理",
    ),
    ("flow_arrow_no_space", "A->B", "A，B"),
    ("flow_arrow_unicode", "配置中心→推理编排→运行时执行", "配置中心，推理编排，运行时执行"),
    (
        "flow_arrow_maas_example",
        "MaaS 主链遵循请求接入 -> 身份与策略判定 -> 域服务处理 -> 推理编排 -> 运行时执行的在线数据面结构。Dashboard 不直接承载单次在线推理请求，而是负责统一配置、统一治理、统一运营和统一展示。",
        "MaaS 主链遵循请求接入，身份与策略判定，域服务处理，推理编排，运行时执行的在线数据面结构。Dashboard 不直接承载单次在线推理请求，而是负责统一配置、统一治理、统一运营和统一展示。",
    ),
    # 5) Preserve embedded titles.
    ("embedded_title", "我喜欢《哈姆雷特》这本书。", "我喜欢《哈姆雷特》这本书。"),
    # 6) Repeated punctuation / social-media noise
    ("noise_qe", "真的假的？？？！！！", "真的假的？！"),
    (
        "noise_ellipsis",
        "这个包把 app.js.map 也发上去了......太离谱了！！！",
        "这个包把 app.js.map 也发上去了。太离谱了！",
    ),
    (
        "noise_ellipsis_cn",
        "【系统提示】请模仿{sad}低沉语气，说“今天下雨了……”",
        "系统提示。请模仿{sad}低沉语气，说“今天下雨了。”",
    ),
    # 7) Spacing: collapse English spaces, remove Chinese spaces, preserve mixed-script boundaries.
    ("english_spaces", "This   is   a   test.", "This is a test."),
    ("chinese_spaces", "这 是　一 段  含有多种空白的文本。", "这是一段含有多种空白的文本。"),
    ("mixed_spaces_1", "这是Anthropic的npm包", "这是 Anthropic 的 npm 包"),
    ("mixed_spaces_2", "今天update到v2.3.1了", "今天 update 到 v2.3.1 了"),
    ("mixed_spaces_3", "处理app.js.map文件", "处理 app.js.map 文件"),
    # 8) Markdown / lists / newlines
    (
        "markdown_link",
        "详情见 [release note](https://github.com/example/release)",
        "详情见 release note https://github.com/example/release",
    ),
    (
        "markdown_heading",
        "# I made a free open source app to help with markdown files",
        "I made a free open source app to help with markdown files",
    ),
    ("list_lines", "- 修复 .map 泄露\n- 发布 v2.3.1", "修复 .map 泄露。发布 v2.3.1"),
    (
        "numbered_lines",
        "1. 安装依赖\n2. 运行测试\n3. 发布 v2.3.1",
        "安装依赖。运行测试。发布 v2.3.1",
    ),
    ("newlines", "第一行\n第二行\n第三行", "第一行。第二行。第三行"),
    # 9) Zero-width characters and idempotence
    ("zero_width_url", "详见 https://x.com/\u200bSafety", "详见 https://x.com/Safety"),
]


@pytest.mark.parametrize("name,text,expected", TEST_CASES, ids=[c[0] for c in TEST_CASES])
def test_normalization_and_idempotence(name, text, expected):
    actual = normalize_tts_text(text)
    assert actual == expected
    assert normalize_tts_text(actual) == actual
