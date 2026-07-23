# spectral_utils.py
"""
Módulo com funções para correção radiométrica, pré-processamento espectral
e otimização de parâmetros para dados hiperespectrais.
Versão completa baseada no Código 1.
"""

import os
import numpy as np
import warnings
import scipy
from scipy.integrate import simpson as simps
from scipy.spatial import ConvexHull
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# CORREÇÃO RADIOMÉTRICA
# -------------------------

def correct_img(img: np.ndarray, dark: np.ndarray, white: np.ndarray = None) -> np.ndarray:
    """
    Aplica correção radiométrica usando dark e white.
    """
    dark = np.mean(dark, axis=0)
    img -= dark
    img_neg_mask = img < 0
    if np.any(img_neg_mask):
        warnings.warn('Imagem com bandas abaixo do dark. Valores negativos truncados para zero.')
        img[img_neg_mask] = 0

    if white is None:
        warnings.warn('White não informado. Retornando imagem apenas com correção de dark.')
        return img

    white = np.mean(white, axis=0)
    white -= dark
    white_neg_mask = white < 0
    if np.any(white_neg_mask):
        warnings.warn('White com bandas abaixo do dark. Valores negativos substituídos por 1e6.')
        white[white_neg_mask] = 1e6

    img /= white
    return img

def get_dark(img_path: str) -> str:
    """Retorna o caminho esperado para o dark referente à imagem."""
    splits = img_path.split('/')
    dark_path = '/'.join(splits[:-1]) + '/DARKREF_' + splits[-1]
    return dark_path

def open_img_normalize(img_path: str, dark_path: str, white_path: str) -> np.ndarray:
    """
    Carrega imagem, dark e white (formato ENVI) e retorna o cubo corrigido.
    """
    from spectral import envi
    img = np.array(envi.open(img_path).load())
    dark = np.array(envi.open(dark_path).load())
    white = np.array(envi.open(white_path).load())
    return correct_img(img, dark, white)

# -------------------------
# PRÉ-PROCESSAMENTO ESPECTRAL
# -------------------------

def snv(spectra: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Standard Normal Variate (SNV).
    """
    spectra = np.atleast_2d(spectra).astype(np.float32, copy=False)
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, ddof=0, keepdims=True)
    small = std < eps
    std_safe = np.where(small, 1.0, std)
    out = (spectra - mean) / std_safe
    if np.any(small):
        out[small[:, 0], :] = spectra[small[:, 0], :] - mean[small[:, 0], :]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def smoothing_filter(spectra: np.ndarray, window_size: int, deriv: int) -> np.ndarray:
    """
    Aplica filtro Savitzky-Golay com suporte para derivadas de ordem superior.
    
    Args:
        spectra: Array de espectros (n_amostras, n_bandas)
        window_size: Tamanho da janela (deve ser ímpar)
        deriv: Ordem da derivada (0, 1, 2, ...)
    
    Returns:
        Array filtrado
    """
    if window_size % 2 == 0:
        raise ValueError("window_size deve ser ímpar")
    
    # Para derivada de ordem > 0, precisamos garantir polyorder >= deriv
    polyorder = max(2, deriv)  # Garante polyorder >= deriv e >= 2 para deriv=2
    
    if window_size <= polyorder:
        raise ValueError(f"window_size ({window_size}) deve ser maior que polyorder ({polyorder})")
    
    signal = np.atleast_2d(spectra)
    coeffs = scipy.signal.savgol_coeffs(
        window_length=window_size,
        polyorder=polyorder,
        deriv=deriv
    ).reshape((1, -1))
    
    return scipy.signal.correlate(signal, coeffs, mode='valid')

def mean_center(
    X: np.ndarray, 
    mean_: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean centering no estilo PLS Toolbox (por variável / banda).
    """
    X = np.atleast_2d(X).astype(np.float32, copy=False)
    
    if mean_ is None:
        mean_ = np.mean(X, axis=0, keepdims=True)
    
    Xc = X - mean_
    return Xc, mean_

def vector_normalization(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Vector normalization (L2 norm) por amostra.
    """
    X = np.atleast_2d(X).astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms

def linear_baseline_correction(X: np.ndarray) -> np.ndarray:
    """
    Linear baseline correction por amostra.
    """
    X = np.atleast_2d(X).astype(np.float32, copy=False)
    n_bands = X.shape[1]
    x = np.arange(n_bands)

    X_corr = np.empty_like(X)

    for i, y in enumerate(X):
        baseline = np.linspace(y[0], y[-1], n_bands)
        X_corr[i] = y - baseline

    return X_corr

def rubberband_baseline_correction(X: np.ndarray) -> np.ndarray:
    """
    Rubberband baseline correction por amostra usando convex hull (implementação clássica).
    """
    X = np.atleast_2d(X).astype(np.float32, copy=False)
    n_bands = X.shape[1]
    x = np.arange(n_bands)

    X_corr = np.empty_like(X)

    for i, y in enumerate(X):
        points = np.column_stack((x, y))
        hull = ConvexHull(points)

        # índices do casco
        hull_idx = hull.vertices

        # ordenar por x (da esquerda para a direita)
        hull_idx = hull_idx[np.argsort(x[hull_idx])]

        # --- selecionar casco inferior ---
        lower_hull = [hull_idx[0]]
        for idx in hull_idx[1:]:
            lower_hull.append(idx)
            while len(lower_hull) >= 3:
                x1, y1 = x[lower_hull[-3]], y[lower_hull[-3]]
                x2, y2 = x[lower_hull[-2]], y[lower_hull[-2]]
                x3, y3 = x[lower_hull[-1]], y[lower_hull[-1]]

                # produto vetorial (orientação)
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

                # se não for parte inferior, remove ponto do meio
                if cross <= 0:
                    lower_hull.pop(-2)
                else:
                    break

        lower_hull = np.array(lower_hull)

        # interpolação da baseline
        baseline = np.interp(x, x[lower_hull], y[lower_hull])
        X_corr[i] = y - baseline

    return X_corr

def preprocess(
    spectra,
    ws: int = 5,
    deriv: int = 0,
    use_snv: bool = True,
    use_linear_baseline: bool = False,
    use_rubberband: bool = False,
    use_vector_norm: bool = False,
    use_mean_center: bool = False,
    mean_: Optional[np.ndarray] = None
):
    """
    Pré-processamento de espectros (ordem fixa):

    SNV
      → Savitzky-Golay (derivadas)
          → Baseline correction (linear ou rubberband)
              → Vector normalization
                  → Mean Center
    """
    X = spectra
    mean_out = None

    # --- SNV ---
    if use_snv:
        X = snv(X)

    # --- Savitzky-Golay ---
    X = smoothing_filter(X, window_size=ws, deriv=deriv)

    # --- Baseline correction ---
    if use_linear_baseline:
        X = linear_baseline_correction(X)

    if use_rubberband:
        X = rubberband_baseline_correction(X)

    # --- Vector normalization ---
    if use_vector_norm:
        X = vector_normalization(X)

    # --- Mean center (sempre por último) ---
    if use_mean_center:
        X, mean_out = mean_center(X, mean_)

    return X, mean_out

# -------------------------
# OTIMIZAÇÃO DE PARÂMETROS
# -------------------------

class GridPreprocessingOptimizeDistance:
    def __init__(self, spectral_data: Dict[str, np.ndarray], 
                 distance_metric: str = "integral", 
                 ws_: List[int] = None, 
                 deriv_: List[int] = None, 
                 use_snv_: List[bool] = None, 
                 slice_list: List[slice] = None,
                 min_window_for_deriv2: int = 7):
        """
        Otimiza parâmetros de pré-processamento incluindo 2ª derivada.
        
        Args:
            spectral_data: Dicionário com {nome_classe: espectros}
            distance_metric: Métrica de distância ("integral")
            ws_: Lista de tamanhos de janela
            deriv_: Lista de ordens de derivada (0, 1, 2)
            use_snv_: Lista de booleanos para uso de SNV
            slice_list: Lista de slices para seleção de bandas
            min_window_for_deriv2: Tamanho mínimo de janela para deriv=2
        """
        self.spectral_data = spectral_data
        self.metrics = {"integral": self.integrate} 
        self.distance_metric = distance_metric
        if self.distance_metric not in self.metrics:
            raise NameError(f"{distance_metric} não disponível! \n Métricas disponíveis: {list(self.metrics.keys())}")
        
        # Configurações padrão que funcionam para deriv=2
        self.ws_ = ws_ if ws_ is not None else [5, 11, 15, 21]
        self.deriv_ = deriv_ if deriv_ is not None else [0, 1, 2]
        self.use_snv_ = use_snv_ if use_snv_ is not None else [False, True]
        self.slice_list = slice_list if slice_list is not None else [slice(None)]
        self.min_window_for_deriv2 = min_window_for_deriv2
        
        # Filtrar combinações inválidas para deriv=2
        self._filter_valid_combinations()
        
        self.grid = {} 
        self.best_avg_distance = 0 
        self.best_grid_params = {}
    
    def _filter_valid_combinations(self):
        """Remove combinações inválidas (ex: ws pequeno para deriv=2)"""
        valid_ws = []
        for ws in self.ws_:
            # Garante que ws seja ímpar
            if ws % 2 == 0:
                ws += 1
            valid_ws.append(ws)
        self.ws_ = list(set(valid_ws))  # Remove duplicatas
        
        # Se deriv=2 está na lista, garante ws mínimo
        if 2 in self.deriv_:
            self.ws_ = [ws for ws in self.ws_ if ws >= self.min_window_for_deriv2]
    
    @staticmethod
    def integrate(preprocessed_data: Dict[str, np.ndarray]) -> Tuple[Dict[Tuple[str, str], float], float]:
        """Calculate the integral of the absolute difference between mean spectra of different classes"""
        classes = list(preprocessed_data.keys())
        
        mean_spectra = {c: np.mean(preprocessed_data[c], axis=0) for c in classes}
        
        grid_distance = {}
        distances = []
        
        class_pairs = [(classes[i], classes[j]) for i in range(len(classes)) 
                      for j in range(len(classes))]
        
        for class1, class2 in class_pairs:
            abs_diff = np.abs(mean_spectra[class1] - mean_spectra[class2])
            area = simps(abs_diff, dx=1.0)
            grid_distance[(class1, class2)] = area
            distances.append(area)
        
        avg_distance = np.mean(distances) if distances else 0
        return grid_distance, avg_distance
    
    def grid_quantify(self):
        """Executa a busca em grade por melhores parâmetros"""
        param_combinations = [(ws, deriv, use_snv, slice_) 
                             for ws in self.ws_ 
                             for deriv in self.deriv_ 
                             for use_snv in self.use_snv_ 
                             for slice_ in self.slice_list]
        
        print(f"Total de combinações a testar: {len(param_combinations)}")
        
        for i, (ws, deriv, use_snv, slice_) in enumerate(param_combinations, 1):
            # Validação adicional
            if deriv == 2 and ws < self.min_window_for_deriv2:
                print(f"Pulando: ws={ws} muito pequeno para deriv=2")
                continue
            
            grid_params = f"ws_{ws}_deriv{deriv}_use_snv{use_snv}_slice{slice_}"
            
            try:
                # Pré-processa os dados
                preprocessed_data = {}
                for k, spectra in self.spectral_data.items():
                    spectra_sliced = spectra[:, slice_]
                    preprocessed_data[k], _ = preprocess(
                        spectra_sliced, 
                        ws=ws, 
                        deriv=deriv, 
                        use_snv=use_snv
                    )
                
                print(f"Testando [{i}/{len(param_combinations)}]: {grid_params}")
                
                # Calcula distância
                grid_distance, avg_distance = self.metrics[self.distance_metric](preprocessed_data)
                
                print(f"  Distância média: {avg_distance:.4f}")
                
                self.grid[grid_params] = {
                    "grid": grid_distance, 
                    "avg_distance": avg_distance
                }
                
                if avg_distance > self.best_avg_distance:
                    self.best_avg_distance = avg_distance
                    self.best_grid_params = {
                        'ws': ws,
                        'deriv': deriv,
                        'use_snv': use_snv,
                        'slice': slice_
                    }
                    
            except Exception as e:
                print(f"Erro com {grid_params}: {e}")
                continue
        
        return self.grid
    
    def get_best_params(self):
        """Get the best preprocessing parameters found"""
        return self.best_grid_params, self.best_avg_distance
    
    def get_detailed_best_params(self):
        """Get detailed best parameters as a dictionary"""
        if not self.best_grid_params:
            print("Nenhum parâmetro ótimo encontrado.")
            return None
        
        ws = self.best_grid_params['ws']
        deriv = self.best_grid_params['deriv']
        use_snv = self.best_grid_params['use_snv']
        slice_ = self.best_grid_params['slice']
        
        grid_params = f"ws_{ws}_deriv{deriv}_use_snv{use_snv}_slice{slice_}"
        
        if grid_params not in self.grid:
            print(f"Parâmetros {grid_params} não encontrados nos resultados.")
            return None
        
        grid_distance = self.grid[grid_params]["grid"]
        avg_distance = self.grid[grid_params]["avg_distance"]
        
        # Plotar espectros médios pré-processados
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        for k in self.spectral_data:
            spectra_sliced = self.spectral_data[k][:, slice_]
            preprocessed_spectra, _ = preprocess(
                spectra_sliced,
                ws=ws,
                deriv=deriv,
                use_snv=use_snv
            )
            plt.plot(np.mean(preprocessed_spectra, axis=0), label=k, alpha=0.8)
        
        deriv_label = {0: "Original", 1: "1ª Derivada", 2: "2ª Derivada"}
        title = f"Melhor: ws={ws}, {deriv_label.get(deriv, f'{deriv}ª Deriv')}, SNV={use_snv}"
        plt.title(title)
        plt.xlabel("Banda")
        plt.ylabel("Intensidade")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plotar heatmap de distâncias
        plt.subplot(1, 2, 2)
        self._plot_distance_heatmap(grid_distance)
        
        plt.suptitle(f"Distância Média: {avg_distance:.4f}", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return {
            'ws': ws,
            'deriv': deriv,
            'use_snv': use_snv,
            'slice': slice_,
            'grid_distance': grid_distance,
            'avg_distance': avg_distance,
            'deriv_label': deriv_label.get(deriv, f'{deriv}ª Derivada')
        }
    
    def _plot_distance_heatmap(self, grid_distance):
        """Plot a heatmap of the pairwise distances between classes"""
        classes = list(set([class_name for pair in grid_distance.keys() for class_name in pair]))
        classes.sort()
        
        n_classes = len(classes)
        distance_matrix = np.zeros((n_classes, n_classes))
        
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes):
                if (class1, class2) in grid_distance:
                    distance_matrix[i, j] = grid_distance[(class1, class2)]
                elif (class2, class1) in grid_distance:
                    distance_matrix[i, j] = grid_distance[(class2, class1)]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            distance_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={'label': 'Distância (Integral da Diferença Absoluta)'}
        )
        plt.title("Matriz de Distâncias entre Classes")
        plt.tight_layout()

# -------------------------
# SALVAMENTO E CARREGAMENTO
# -------------------------

def save_best_preprocessing_parameters(params: dict, save_path):
    """
    Salva os melhores parâmetros de pré-processamento em JSON.
    """
    os.makedirs(save_path, exist_ok=True)
    json_preprocessing_parameters_path = os.path.join(save_path, "preprocessing_params.json")
    
    def convert_for_json(obj):
        if isinstance(obj, slice):
            return {
                '__type__': 'slice',
                'start': obj.start,
                'stop': obj.stop,
                'step': obj.step
            }
        elif isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    params_converted = convert_for_json(params)
    
    with open(json_preprocessing_parameters_path, "w") as f:
        json.dump(params_converted, f, indent=4)
    
    print(f"Parâmetros salvos em: {json_preprocessing_parameters_path}")
    return json_preprocessing_parameters_path

def load_json_from_path(json_path) -> dict:
    """
    Carrega um JSON de caminho 'json_path'.
    """
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data

# -------------------------
# EXEMPLO DE USO (OPCIONAL)
# -------------------------
if __name__ == "__main__":
    # Exemplo de uso com dados simulados
    np.random.seed(42)
    n_samples = 100
    n_bands = 200
    
    spectral_data = {
        'Classe_A': np.random.randn(n_samples, n_bands) * 0.5 + np.sin(np.linspace(0, 10, n_bands)),
        'Classe_B': np.random.randn(n_samples, n_bands) * 0.5 + np.cos(np.linspace(0, 10, n_bands)),
        'Classe_C': np.random.randn(n_samples, n_bands) * 0.5 + np.linspace(0, 1, n_bands)
    }
    
    optimizer = GridPreprocessingOptimizeDistance(
        spectral_data=spectral_data,
        ws_=[5, 7, 9, 11, 15, 21],
        deriv_=[0, 1, 2],
        use_snv_=[True, False],
        slice_list=[slice(None), slice(50, 150)],
        min_window_for_deriv2=7
    )
    
    print("Iniciando otimização de parâmetros...")
    results = optimizer.grid_quantify()
    
    best_params, best_distance = optimizer.get_best_params()
    print(f"\nMelhores parâmetros encontrados:")
    print(f"  window_size: {best_params['ws']}")
    print(f"  derivada: {best_params['deriv']}ª ordem")
    print(f"  SNV: {best_params['use_snv']}")
    print(f"  slice: {best_params['slice']}")
    print(f"  Distância média: {best_distance:.4f}")
    
    detailed = optimizer.get_detailed_best_params()
    
    if detailed:
        save_path = "./results"
        save_best_preprocessing_parameters(detailed, save_path)