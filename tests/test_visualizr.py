from fastapi.testclient import TestClient

from visualizr.gui import app_block

app = app_block()

def test_visualizr():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200