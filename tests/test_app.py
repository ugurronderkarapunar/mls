# Streamlit entegrasyon testi (basit import testi)
def test_app_imports():
    try:
        from app import app
        assert True
    except Exception as e:
        assert False, f"Uygulama import hatası: {e}"
