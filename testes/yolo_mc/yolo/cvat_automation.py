import sys
from pathlib import Path
from urllib.parse import urlparse

# tenta importar o SDK
try:
    from cvat_sdk import make_client
    from cvat_sdk.core.exceptions import InvalidHostException
except ImportError:
    print("[ERRO] cvat-sdk não está instalado. Rode: pip install cvat-sdk")
    sys.exit(1)

# ===== CONFIGURAÇÃO =====
CVAT_URL = "http://localhost:8080"
USER = "admin"
PASS = "sua_senha_criada_no_docker"
# =========================

def test_connection():
    print(f"[INFO] Testando conexão com {CVAT_URL} ...")
    try:
        # Tenta com make_client usando host e port
        parsed = urlparse(CVAT_URL)
        host = parsed.hostname
        port = parsed.port
        if not port:
            port = 443 if parsed.scheme == "https" else 80

        print(f"[INFO] Tentando make_client(host='{host}', port={port})...")
        with make_client(host=host, port=port, credentials=(USER, PASS)) as client:
            # Testa a API
            projects = client.projects.list()
            print(f"[OK] Conexão bem-sucedida! Projetos encontrados: {len(projects)}")
            return True
    except InvalidHostException as e:
        print(f"[ERRO] Conexão falhou: {e}")
        print("   Dica: O servidor pode não estar rodando ou a URL está incorreta.")
        print("   Verifique se http://localhost:8080 está acessível no navegador.")
    except Exception as e:
        print(f"[ERRO] Falha inesperada: {e}")
        # Tenta uma abordagem alternativa: usar a classe Client com URL
        try:
            from cvat_sdk import Client
            print("[INFO] Tentando Client(url=CVAT_URL, auth=(USER, PASS)) ...")
            with Client(url=CVAT_URL, auth=(USER, PASS)) as client:
                projects = client.projects.list()
                print(f"[OK] Conexão via Client funcionou! Projetos: {len(projects)}")
                return True
        except Exception as e2:
            print(f"[ERRO] Também falhou com Client: {e2}")

    print("\n[AVISO] Não foi possível conectar ao CVAT.")
    print("   Soluções:")
    print("   1. Certifique-se de que o Docker está rodando: 'docker-compose ps'")
    print("   2. Verifique a URL e as credenciais.")
    print("   3. Tente desabilitar firewall temporariamente.")
    return False

if __name__ == "__main__":
    test_connection()