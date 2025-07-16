import movie_agent.framepack as framepack


def test_generate_video(monkeypatch):
    calls = {}

    class DummyClient:
        def __init__(self, url):
            calls['url'] = url

        def predict(self, frames_dir, fps, output, api_name=None):
            calls['args'] = (frames_dir, fps, output)
            calls['api_name'] = api_name
            return 'result-data'

    monkeypatch.setattr(framepack, 'Client', DummyClient)
    monkeypatch.setattr(framepack, 'FRAMEPACK_HOST', '1.2.3.4')
    monkeypatch.setattr(framepack, 'FRAMEPACK_PORT', '1234')
    monkeypatch.setattr(framepack, 'FRAMEPACK_API_NAME', '/validate_and_process')

    result = framepack.generate_video('frames', fps=30, output='out.mp4')

    assert result == 'result-data'
    assert calls['url'] == 'http://1.2.3.4:1234/'
    assert calls['args'] == ('frames', 30, 'out.mp4')
    assert calls['api_name'] == framepack.FRAMEPACK_API_NAME
