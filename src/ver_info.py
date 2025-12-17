import joblib
import os

def ver_columnas():
    # 1. Calculamos la ruta exacta para que no falle
    # Obtenemos la carpeta donde está este archivo (src)
    carpeta_actual = os.path.dirname(os.path.abspath(__file__))
    # Subimos un nivel y entramos a notebooks/pkl
    ruta_pkl = os.path.join(carpeta_actual, '..', 'notebooks', 'pkl')

    nombres = ["pulmon", "mama", "prostata", "gastrico", "cervical"]

    print(f"📂 Buscando archivos en: {ruta_pkl}")
    print("=" * 50)

    for n in nombres:
        archivo = os.path.join(ruta_pkl, f'columnas_{n}.pkl')
        try:
            # Cargamos las columnas
            cols = joblib.load(archivo)
            
            print(f"\n🧬 {n.upper()} - Necesita estas {len(cols)} columnas:")
            print("-" * 20)
            # Imprimimos la lista tal cual para que la copies
            print(cols) 
            print("-" * 20)
            
        except FileNotFoundError:
            print(f"\n❌ NO ENCONTRADO: {archivo}")
            print("Revisa que el nombre del archivo o la carpeta sean correctos.")
        except Exception as e:
            print(f"\n❌ Error cargando {n}: {e}")

if __name__ == "__main__":
    ver_columnas()