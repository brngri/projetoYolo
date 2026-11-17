import os
import sys
import site
from pathlib import Path

def patch_labelimg_canvas():
    # Encontra a pasta de instalação do labelImg
    site_packages = site.getsitepackages()
    canvas_path = None
    
    for sp in site_packages:
        test_path = Path(sp) / "libs" / "canvas.py"
        if test_path.exists():
            canvas_path = test_path
            break
    
    if not canvas_path:
        print("[ERRO] Não encontrei o arquivo libs/canvas.py do LabelImg.")
        print("Tentando localizar manualmente...")
        # Tenta procurar em todos os diretórios do sys.path
        for p in sys.path:
            test_path = Path(p) / "libs" / "canvas.py"
            if test_path.exists():
                canvas_path = test_path
                break
    
    if not canvas_path:
        print("[ERRO] Não foi possível localizar o canvas.py.")
        print("Tente reinstalar o labelImg: pip install labelImg==1.8.6")
        return False
    
    print(f"[INFO] Arquivo encontrado: {canvas_path}")
    
    # Lê o conteúdo
    with open(canvas_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # A linha problemática (geralmente na função paintEvent)
    # Procura por "drawLine(self.prev_point.x()" e adiciona int()
    old_pattern = "p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())"
    new_pattern = "p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), self.pixmap.height())"
    
    if old_pattern in content:
        new_content = content.replace(old_pattern, new_pattern)
        with open(canvas_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("[OK] Patch aplicado com sucesso! O LabelImg agora deve funcionar.")
        return True
    else:
        # Tenta uma variação (pode ter espaços diferentes)
        import re
        pattern = r"p\.drawLine\(self\.prev_point\.x\(\)\s*,\s*0\s*,\s*self\.prev_point\.x\(\)\s*,\s*self\.pixmap\.height\(\)\)"
        if re.search(pattern, content):
            new_content = re.sub(pattern, "p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), self.pixmap.height())", content)
            with open(canvas_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("[OK] Patch aplicado com sucesso (regex)!")
            return True
        else:
            print("[AVISO] Não encontrei a linha exata. O LabelImg pode já estar corrigido.")
            return False

if __name__ == "__main__":
    patch_labelimg_canvas()