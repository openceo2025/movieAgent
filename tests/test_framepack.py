import movie_agent.framepack as framepack


def test_generate_video(monkeypatch):
    calls = {}

    class DummyClient:
        def __init__(self, url):
            calls['url'] = url

        def predict(self, *args, api_name=None):
            calls['args'] = args
            calls['api_name'] = api_name
            return 'result-data'

    def fake_handle_file(path):
        calls['handled'] = path
        return {'path': path}

    monkeypatch.setattr(framepack, 'Client', DummyClient)
    monkeypatch.setattr(framepack, 'handle_file', fake_handle_file)
    monkeypatch.setattr(framepack, 'FRAMEPACK_HOST', '1.2.3.4')
    monkeypatch.setattr(framepack, 'FRAMEPACK_PORT', '1234')
    monkeypatch.setattr(framepack, 'FRAMEPACK_API_NAME', '/validate_and_process')
    monkeypatch.setattr(framepack, 'FRAMEPACK_FN_INDEX', 1)
    monkeypatch.setattr(framepack, 'FRAMEPACK_FN_INDEX', 1)

    result = framepack.generate_video(
        'start.png',
        prompt='p',
        seed=1,
        video_length=2,
        latent_window_size=3,
        steps=4,
        cfg=5.0,
        gs=6.0,
        rs=7.0,
        gpu_memory_preservation=8.0,
        use_teacache=True,
        mp4_crf=9,
    )

    assert result == 'result-data'
    assert calls['url'] == 'http://1.2.3.4:1234/'
    assert calls['handled'] == 'start.png'
    assert calls['args'] == (
        {'path': 'start.png'},
        'p',
        '',
        1,
        2,
        3,
        4,
        5.0,
        6.0,
        7.0,
        8.0,
        True,
        9,
    )
    assert calls['api_name'] == framepack.FRAMEPACK_API_NAME


def test_generate_video_debug(monkeypatch, capsys):
    calls = {}

    class DummyClient:
        def __init__(self, url):
            calls['url'] = url

        def predict(self, *args, api_name=None):
            calls['args'] = args
            calls['api_name'] = api_name
            return 'result-data'

    def fake_handle_file(path):
        calls['handled'] = path
        return {'path': path}

    monkeypatch.setattr(framepack, 'Client', DummyClient)
    monkeypatch.setattr(framepack, 'handle_file', fake_handle_file)
    monkeypatch.setattr(framepack, 'FRAMEPACK_HOST', '1.2.3.4')
    monkeypatch.setattr(framepack, 'FRAMEPACK_PORT', '1234')
    monkeypatch.setattr(framepack, 'FRAMEPACK_API_NAME', '/validate_and_process')
    monkeypatch.setattr(framepack, 'FRAMEPACK_FN_INDEX', 1)
    result = framepack.generate_video(
        'start.png',
        prompt='p',
        seed=1,
        video_length=2,
        latent_window_size=3,
        steps=4,
        cfg=5.0,
        gs=6.0,
        rs=7.0,
        gpu_memory_preservation=8.0,
        use_teacache=True,
        mp4_crf=9,
        debug=True,
    )

    out = capsys.readouterr().out.strip().splitlines()

    assert result == 'result-data'
    assert out[0] == '[DEBUG] framepack url: http://1.2.3.4:1234//validate_and_process'
    assert out[1].startswith('[DEBUG] framepack params:')
    assert "'mp4_crf': 9" in out[1]
    assert "'prompt': 'p'" in out[1]
    assert "'use_teacache': True" in out[1]
    assert out[2] == '[DEBUG] framepack response: result-data'


def test_generate_video_fallback(monkeypatch):
    calls = []

    class DummyClient:
        def __init__(self, url):
            pass

        def predict(self, *args, api_name=None, fn_index=None):
            calls.append(api_name if api_name is not None else fn_index)
            if api_name == '/validate_and_process':
                raise Exception('fail')
            return 'ok'

    monkeypatch.setattr(framepack, 'Client', DummyClient)
    monkeypatch.setattr(framepack, 'handle_file', lambda p: {'path': p})
    monkeypatch.setattr(framepack, 'FRAMEPACK_HOST', '1.2.3.4')
    monkeypatch.setattr(framepack, 'FRAMEPACK_PORT', '1234')
    monkeypatch.setattr(framepack, 'FRAMEPACK_API_NAME', '/validate_and_process')
    monkeypatch.setattr(framepack, 'FRAMEPACK_FN_INDEX', 1)
    result = framepack.generate_video(
        'start.png',
        prompt='p',
        seed=1,
        video_length=2,
        latent_window_size=3,
        steps=4,
        cfg=5.0,
        gs=6.0,
        rs=7.0,
        gpu_memory_preservation=8.0,
        use_teacache=True,
        mp4_crf=9,
    )

    assert result == 'ok'
    assert calls == ['/validate_and_process', '/predict']


def test_generate_video_fallback_fn_index(monkeypatch):
    calls = []

    class DummyClient:
        def __init__(self, url):
            pass

        def predict(self, *args, api_name=None, fn_index=None):
            calls.append(api_name if api_name is not None else fn_index)
            raise Exception('fail')

    monkeypatch.setattr(framepack, 'Client', DummyClient)
    monkeypatch.setattr(framepack, 'handle_file', lambda p: {'path': p})
    monkeypatch.setattr(framepack, 'FRAMEPACK_HOST', '1.2.3.4')
    monkeypatch.setattr(framepack, 'FRAMEPACK_PORT', '1234')
    monkeypatch.setattr(framepack, 'FRAMEPACK_API_NAME', '/validate_and_process')
    monkeypatch.setattr(framepack, 'FRAMEPACK_FN_INDEX', 1)
    result = framepack.generate_video(
        'start.png',
        prompt='p',
        seed=1,
        video_length=2,
        latent_window_size=3,
        steps=4,
        cfg=5.0,
        gs=6.0,
        rs=7.0,
        gpu_memory_preservation=8.0,
        use_teacache=True,
        mp4_crf=9,
    )

    assert result is None
    assert calls == ['/validate_and_process', '/predict', 1]
