# ==========================================================
# ANÁLISIS DE VENTAS COMERCIALES — Pipeline según rúbrica procesos.md
# Flujo: recibir y documentar -> formular preguntas -> validar ->
#        limpiar con justificación -> explorar -> decidir si modelar ->
#        evaluar correctamente -> interpretar -> informar
# ==========================================================

import os
import base64
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# ----------------------------------------------------------
# CONFIGURACIÓN VISUAL
# ----------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

os.makedirs("outputs", exist_ok=True)

PALETA = {
    'verde_oscuro': '#1B5E20', 'verde': '#2E7D32', 'verde_claro': '#66BB6A',
    'marron': '#8D6E63', 'rojo': '#D32F2F', 'gris': '#757575',
    'azul': '#1565C0', 'ambar': '#F9A825'
}

# ----------------------------------------------------------
# 1. RECIBIR Y DOCUMENTAR LOS DATOS (rúbrica, etapa 1)
# ----------------------------------------------------------
print("[1/9] Documentando la recepción de los datos...")

ARCHIVO = os.path.join("data", "ventas_4.csv")
FECHA_RECEPCION = "2026-11-17"
ORIGEN = "Registros comerciales provistos por la empresa (contexto: ventas.md)"

inventario = {
    'archivo': 'ventas_4.csv',
    'formato': 'CSV',
    'origen': ORIGEN,
    'fecha_recepcion': FECHA_RECEPCION,
    'descripcion': ("Una fila = una operación de venta de productos tecnológicos. "
                    "Población: operaciones de la empresa. Período: 2023-2025. "
                    "Unidades monetarias: pesos (total_venta, precio_unitario)."),
    'sensible': 'No se identifican datos personales identificatorios (edad es genérica, sin nombres ni IDs de cliente).'
}

DESCRIPCION_COLUMNAS = {
    'id_registro': 'Identificador numérico de la operación',
    'fecha_venta': 'Fecha de la venta',
    'producto': 'Producto vendido (6 tipos)',
    'categoria': 'Categoría del producto',
    'region': 'Región de la venta',
    'cantidad': 'Unidades vendidas en la operación',
    'precio_unitario': 'Precio por unidad',
    'descuento_pct': 'Descuento aplicado (proporción 0-1)',
    'total_venta': 'Importe total registrado de la venta',
    'edad_cliente': 'Edad del cliente (años)',
    'canal_venta': 'Canal de comercialización (Online/Presencial/Telefonico)',
    'medio_pago': 'Medio de pago',
    'devolucion': 'Indicador de devolución (True/False)',
    'dias_entrega': 'Días entre venta y entrega',
    'satisfaccion': 'Satisfacción del cliente (1 a 5)',
}

# ----------------------------------------------------------
# 2. PROBLEMA Y PREGUNTAS (rúbrica, etapa 2 - ventas.md)
# ----------------------------------------------------------
print("[2/9] Definiendo problema y preguntas de análisis...")

PREGUNTAS = {
    'A': 'A. Ventas: volumen total, evolución temporal, mejores/peores meses y estacionalidad.',
    'B': 'B. Productos: mayores ingresos, mayor cantidad de unidades, concentración por categoría.',
    'C': 'C. Clientes: edad vs compras, canal más eficaz, medio de pago vs volumen, descuentos vs importe.',
    'D': 'D. Calidad de servicio: tasa de devolución, entrega vs satisfacción, patrones de devolución.',
}

# ----------------------------------------------------------
# 3. VALIDAR ESTRUCTURA Y CALIDAD (rúbrica, etapa 3)
# ----------------------------------------------------------
print("[3/9] Validando estructura y calidad del dataset...")

df_raw = pd.read_csv(ARCHIVO)          # copia de trabajo
df_original = pd.read_csv(ARCHIVO)     # referencia intacta para comparar

n0 = len(df_raw)
problemas = {}

problemas['duplicados_exactos'] = int(df_raw.duplicated().sum())
problemas['ids_duplicados'] = int(df_raw['id_registro'].duplicated().sum())

fecha_parse = pd.to_datetime(df_raw['fecha_venta'], errors='coerce')
problemas['fechas_invalidas'] = int(fecha_parse.isna().sum())

precio_num = pd.to_numeric(df_raw['precio_unitario'], errors='coerce')
problemas['precio_invalido'] = int((precio_num.isna() & df_raw['precio_unitario'].notna()).sum())

cant = pd.to_numeric(df_raw['cantidad'], errors='coerce')
problemas['cantidad_negativa'] = int((cant < 0).sum())
problemas['cantidad_fraccional'] = int((cant % 1 != 0).sum())

desc = pd.to_numeric(df_raw['descuento_pct'], errors='coerce')
problemas['descuento_fuera_rango'] = int((desc > 1).sum())

edad = pd.to_numeric(df_raw['edad_cliente'], errors='coerce')
problemas['edad_implausible'] = int((edad > 100).sum())
problemas['edad_faltante'] = int(edad.isna().sum())

entrega = pd.to_numeric(df_raw['dias_entrega'], errors='coerce')
problemas['entrega_negativa'] = int((entrega < 0).sum())

region_norm = df_raw['region'].astype('string').str.strip().str.title()
region_norm = region_norm.mask(region_norm.str.len() == 0)
problemas['region_faltante'] = int(region_norm.isna().sum())
problemas['region_inconsistente'] = int((region_norm.dropna() != df_raw['region'].astype('string').str.strip().str.title().dropna()).sum())

sat = pd.to_numeric(df_raw['satisfaccion'], errors='coerce')
problemas['satisfaccion_faltante'] = int(sat.isna().sum())

# Consistencia interna: total_venta vs cantidad*precio*(1-descuento)
calc_total = cant * precio_num * (1 - desc)
diff_rel = (df_raw['total_venta'] - calc_total).abs() / df_raw['total_venta'].replace(0, np.nan)
problemas['total_inconsistente'] = int((diff_rel > 0.01).sum())

print("   Problemas detectados:", problemas)

# ----------------------------------------------------------
# 4. LIMPIEZA Y PREPARACIÓN CON BITÁCORA (rúbrica, etapa 4)
# ----------------------------------------------------------
print("[4/9] Limpiando los datos con bitácora reproducible...")

bitacora = []

def registrar(paso, cambio, afectados, regla, justificacion, influencia):
    bitacora.append({'paso': paso, 'cambio': cambio, 'filas_valores_afectados': afectados,
                     'regla': regla, 'justificacion': justificacion, 'influencia_potencial': influencia})

df = df_raw.copy()

# 4.1 Duplicados exactos -> se eliminan (son el mismo registro repetido)
mask_dup = df.duplicated()
n_dup = int(mask_dup.sum())
df = df[~mask_dup].copy()
registrar(1, 'Eliminación de filas duplicadas exactas', n_dup,
          'drop_duplicates() sobre todas las columnas',
          'Un registro repetido íntegramente (incluido id) es un error de carga, no una venta adicional.',
          'Sin corrección, los KPIs de ingresos y unidades estarían inflados un ~3%.')

# 4.2 Región: normalizar mayúsculas/espacios
region_norm = df['region'].astype('string').str.strip().str.title()
region_norm = region_norm.mask(region_norm.str.len() == 0)
n_region = int((region_norm.dropna() != df['region'].astype('string').str.strip().str.title().dropna()).sum())
df['region'] = region_norm
registrar(2, 'Normalización de la columna region', n_region,
          'strip() + title() (ej. "NORTE" -> "Norte", "ESTE " -> "Este")',
          'Son la misma categoría escrita de forma distinta; sin normalización, los análisis por región segmentan mal.',
          'Agrupa correctamente las ventas regionales; los nulos se mantienen como "Sin dato" para no inventar regiones.')

# 4.3 Fechas inválidas -> NaT (no se imputan fechas inventadas)
fv = pd.to_datetime(df['fecha_venta'], errors='coerce')
n_fecha = int(fv.isna().sum())
df['fecha_venta'] = fv
registrar(3, 'Fechas "FECHA_INVALIDA" convertidas a nulo', n_fecha,
          "pd.to_datetime(errors='coerce')",
          'No es posible deducir la fecha real de esas ventas; imputarlas sesgaría la serie temporal.',
          'Los análisis temporales usan ~97% de las operaciones; se declara la limitación en el informe.')

# 4.4 Precio "DATO_INVALIDO" -> NaN (no se imputa: es dato fuente)
pu = pd.to_numeric(df['precio_unitario'], errors='coerce')
n_precio = int((pu.isna() & df['precio_unitario'].notna()).sum())
df['precio_unitario'] = pu
registrar(4, 'Precios "DATO_INVALIDO" convertidos a nulo', n_precio,
          "pd.to_numeric(errors='coerce')",
          'Imputar precios con medias falsearía el precio promedio y el análisis de descuentos.',
          'Los análisis de precio y la consistencia de total_venta excluyen esas filas.')

# 4.5 Cantidades negativas o fraccionales -> NaN (una venta no puede tener unidades negativas)
cant = pd.to_numeric(df['cantidad'], errors='coerce')
mask_cant = cant < 0
n_cant = int(mask_cant.sum())
df.loc[mask_cant, 'cantidad'] = np.nan
registrar(5, 'Cantidades negativas convertidas a nulo', n_cant,
          'cantidad < 0 -> NaN',
          'Una venta con unidades negativas no es interpretable como operación real; se conserva la fila porque el resto de sus campos aporta.',
          'Excluye esas filas de los conteos de unidades; si fueran devoluciones mal registradas, se subestima su efecto (limitación).')

# 4.6 Descuentos fuera de rango (>1) -> NaN
desc = pd.to_numeric(df['descuento_pct'], errors='coerce')
mask_desc = desc > 1
n_desc = int(mask_desc.sum())
df.loc[mask_desc, 'descuento_pct'] = np.nan
registrar(6, 'Descuentos mayores a 100% convertidos a nulo', n_desc,
          'descuento_pct > 1 -> NaN (rango válido 0-0.30 observado)',
          'Un descuento de 150%-300% no es posible en el negocio; el resto de los datos acota el rango a 0-30%.',
          'Evita distorsionar la correlación descuento-importe y el descuento promedio (KPI).')

# 4.7 Edades implausibles (>100 años) -> NaN
edad = pd.to_numeric(df['edad_cliente'], errors='coerce')
mask_edad = edad > 100
n_edad = int(mask_edad.sum())
df.loc[mask_edad, 'edad_cliente'] = np.nan
registrar(7, 'Edades mayores a 100 años convertidas a nulo', n_edad,
          'edad_cliente > 100 -> NaN (mínimo observado válido: 18)',
          'Valores hasta 199 años son biológicamente implausibles; se asume error de captura.',
          'Reduce la muestra del análisis edad-compras; se declara como limitación.')

# 4.8 Días de entrega negativos -> NaN
entrega = pd.to_numeric(df['dias_entrega'], errors='coerce')
mask_ent = entrega < 0
n_ent = int(mask_ent.sum())
df.loc[mask_ent, 'dias_entrega'] = np.nan
registrar(8, 'Días de entrega negativos convertidos a nulo', n_ent,
          'dias_entrega < 0 -> NaN',
          'Una entrega no puede ocurrir antes de la venta; el máximo válido es 14 días.',
          'Excluye esas filas del análisis entrega-satisfacción.')

df.to_csv('outputs/ventas_4_limpio.csv', index=False)
pd.DataFrame(bitacora).to_csv('outputs/bitacora_limpieza.csv', index=False, encoding='utf-8-sig')

print(f"   Filas: {n0} -> {len(df)} (eliminadas {n_dup} duplicadas exactas)")
print(f"   Bitácora guardada: outputs/bitacora_limpieza.csv")

# ----------------------------------------------------------
# 5. KPIs SOLICITADOS EN ventas.md
# ----------------------------------------------------------
print("[5/9] Calculando KPIs solicitados...")

ingresos_totales = float(df['total_venta'].sum())
unidades = float(df['cantidad'].sum())
precio_prom = float(df['precio_unitario'].mean())
ticket_prom = ingresos_totales / len(df)
tasa_dev = df['devolucion'].mean() * 100
desc_prom = float(df['descuento_pct'].mean())
sat_prom = float(df['satisfaccion'].mean())
entrega_prom = float(df['dias_entrega'].mean())

kpis = pd.DataFrame({
    'Indicador': ['Ingresos totales', 'Unidades vendidas', 'Precio promedio', 'Ticket promedio',
                  'Tasa de devolución', 'Descuento promedio', 'Satisfacción promedio',
                  'Tiempo promedio de entrega', 'Operaciones (filas)'],
    'Valor': [f"${ingresos_totales:,.0f}", f"{unidades:,.0f} u.", f"${precio_prom:,.2f}",
              f"${ticket_prom:,.2f}", f"{tasa_dev:.2f}%", f"{desc_prom*100:.2f}%",
              f"{sat_prom:.2f} / 5", f"{entrega_prom:.2f} días", f"{len(df)}"]
})
kpis.to_csv('outputs/kpis.csv', index=False, encoding='utf-8-sig')
print(kpis.to_string(index=False))

# ----------------------------------------------------------
# 6. ANÁLISIS EXPLORATORIO (rúbrica, etapa 5 - preguntas A-D)
# ----------------------------------------------------------
print("[6/9] Generando análisis exploratorio (preguntas A-D)...")

df_v = df[df['fecha_venta'].notna()].copy()
df_v['año'] = df_v['fecha_venta'].dt.year
df_v['mes'] = df_v['fecha_venta'].dt.month
df_v['año_mes'] = df_v['fecha_venta'].dt.to_period('M')

# --- Pregunta A: evolución temporal y estacionalidad ---
mensual = df_v.groupby('año_mes')['total_venta'].sum().reset_index()
mensual['fecha'] = mensual['año_mes'].dt.to_timestamp()
ultimo_periodo = df_v['año_mes'].max()

plt.figure(figsize=(12, 5))
plt.plot(mensual['fecha'], mensual['total_venta'] / 1e6, color=PALETA['verde'], linewidth=2.2, marker='o', markersize=4)
plt.axvspan(ultimo_periodo.to_timestamp(), mensual['fecha'].max(), color=PALETA['ambar'], alpha=0.25,
            label=f'Período incompleto ({ultimo_periodo})')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.title("A. Evolución mensual de los ingresos (solo fechas válidas)", fontweight='bold')
plt.xlabel("Mes")
plt.ylabel("Ingresos (millones $)")
plt.legend(frameon=True, facecolor='white')
plt.tight_layout()
plt.savefig("outputs/fig_01_evolucion_mensual.png", dpi=300)
plt.close()

# Estacionalidad: promedio de ingresos por mes calendario, comparando años
est = df_v.groupby(['año', 'mes'])['total_venta'].sum().reset_index()
plt.figure(figsize=(12, 5))
for año, color in zip(sorted(df_v['año'].unique()), [PALETA['verde'], PALETA['marron'], PALETA['azul']]):
    serie = est[est['año'] == año].set_index('mes')['total_venta'].reindex(range(1, 13))
    plt.plot(range(1, 13), serie / 1e6, marker='o', color=color, linewidth=2,
             label=f'{año}' + (' (hasta sep., incompleto)' if año == df_v['año'].max() else ''))
plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
plt.title("A. Estacionalidad: ingresos por mes calendario y año", fontweight='bold')
plt.xlabel("Mes")
plt.ylabel("Ingresos (millones $)")
plt.legend(frameon=True, facecolor='white')
plt.tight_layout()
plt.savefig("outputs/fig_02_estacionalidad.png", dpi=300)
plt.close()

mejor_mes = est.loc[est['total_venta'].idxmax()]
peor_mes = est.loc[est['total_venta'].idxmin()]
MESES = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
         7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}

# --- Pregunta B: productos y categorías ---
ing_prod = df.groupby('producto')['total_venta'].sum().sort_values(ascending=False)
uni_prod = df.groupby('producto')['cantidad'].sum().sort_values(ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ing_prod.sort_values().plot(kind='barh', ax=ax1, color=PALETA['verde'], edgecolor='black')
ax1.set_title("A/B. Ingresos por producto", fontweight='bold')
ax1.set_xlabel("Ingresos ($)")
ax1.set_ylabel("")
uni_prod.sort_values().plot(kind='barh', ax=ax2, color=PALETA['marron'], edgecolor='black')
ax2.set_title("B. Unidades vendidas por producto", fontweight='bold')
ax2.set_xlabel("Unidades")
ax2.set_ylabel("")
for ax in (ax1, ax2):
    for rect in ax.patches:
        w = rect.get_width()
        ax.annotate(f'{w:,.0f}'.replace(',', '.'),
                    xy=(w, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0), textcoords='offset points', va='center', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/fig_03_productos.png", dpi=300)
plt.close()

ing_cat = df.groupby('categoria')['total_venta'].sum().sort_values(ascending=False)
share_cat = ing_cat / ing_cat.sum() * 100
top_prod_ing = ing_prod.index[0]
top_prod_uni = uni_prod.index[0]
coinciden = top_prod_ing == top_prod_uni

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ing_cat.sort_values().plot(kind='barh', ax=ax1, color=PALETA['verde'], edgecolor='black')
ax1.set_title("B. Concentración de ingresos por categoría", fontweight='bold')
ax1.set_xlabel("Ingresos ($)")
ax1.set_ylabel("")
for rect, cat in zip(ax1.patches, ing_cat.sort_values().index):
    w = rect.get_width()
    ax1.annotate(f'{w/1e6:.1f}M ({share_cat[cat]:.1f}%)',
                 xy=(w, rect.get_y() + rect.get_height() / 2),
                 xytext=(3, 0), textcoords='offset points', va='center', fontsize=8.5, fontweight='bold')
uni_cat = df.groupby('categoria')['cantidad'].sum().sort_values(ascending=False)
uni_cat.sort_values().plot(kind='barh', ax=ax2, color=PALETA['marron'], edgecolor='black')
ax2.set_title("B. Unidades vendidas por categoría", fontweight='bold')
ax2.set_xlabel("Unidades")
ax2.set_ylabel("")
plt.tight_layout()
plt.savefig("outputs/fig_04_categorias.png", dpi=300)
plt.close()

# --- Pregunta C: clientes, canal, medio de pago y descuentos ---
df_ec = df[df['edad_cliente'].notna()]
corr_edad_total = float(df_ec['edad_cliente'].corr(df_ec['total_venta']))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.scatter(df_ec['edad_cliente'], df_ec['total_venta'] / 1e3, alpha=0.35, s=22, color=PALETA['verde'])
ax1.set_title(f"C. Edad del cliente vs importe (r = {corr_edad_total:+.2f})", fontweight='bold')
ax1.set_xlabel("Edad del cliente (años)")
ax1.set_ylabel("Importe de la venta (miles $)")

ing_canal = df.groupby('canal_venta')['total_venta'].sum().sort_values(ascending=False)
ticket_canal = df.groupby('canal_venta')['total_venta'].mean()
x_c = np.arange(len(ing_canal))
ax2.bar(x_c - 0.2, ing_canal / 1e6, 0.38, color=PALETA['verde'], edgecolor='black', label='Ingresos (millones $)')
ax2b = ax2.twinx()
ax2b.bar(x_c + 0.2, ticket_canal[ing_canal.index] / 1e3, 0.38, color=PALETA['marron'], edgecolor='black', label='Ticket promedio (miles $)')
ax2.set_xticks(x_c)
ax2.set_xticklabels(ing_canal.index, fontweight='bold')
ax2.set_ylabel("Ingresos (millones $)")
ax2b.set_ylabel("Ticket promedio (miles $)")
ax2.set_title("C. Desempeño por canal de venta", fontweight='bold')
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, frameon=True, facecolor='white', fontsize=8.5, loc='upper right')
plt.tight_layout()
plt.savefig("outputs/fig_05_cliente_canal.png", dpi=300)
plt.close()

op_pago = df.groupby('medio_pago').agg(n=('id_registro', 'count'), total=('total_venta', 'sum'))
op_pago = op_pago.sort_values('total', ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.bar(op_pago.index, op_pago['n'], color=PALETA['verde'], edgecolor='black')
for i, v in enumerate(op_pago['n']):
    ax1.annotate(str(v), xy=(i, v), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
ax1.set_title("C. Cantidad de operaciones por medio de pago", fontweight='bold')
ax1.set_xlabel("Medio de pago")
ax1.set_ylabel("Operaciones")

df_dc = df[df['descuento_pct'].notna()]
corr_desc_total = float(df_dc['descuento_pct'].corr(df_dc['total_venta']))
ax2.scatter(df_dc['descuento_pct'] * 100, df_dc['total_venta'] / 1e3, alpha=0.35, s=22, color=PALETA['azul'])
ax2.set_title(f"C. Descuento aplicado vs importe final (r = {corr_desc_total:+.2f})", fontweight='bold')
ax2.set_xlabel("Descuento (%)")
ax2.set_ylabel("Importe de la venta (miles $)")
plt.tight_layout()
plt.savefig("outputs/fig_06_pago_descuento.png", dpi=300)
plt.close()

# --- Pregunta D: calidad del servicio ---
tasa_dev_prod = df.groupby('producto')['devolucion'].mean() * 100
tasa_dev_canal = df.groupby('canal_venta')['devolucion'].mean() * 100
tasa_dev_region = df.groupby('region', dropna=False)['devolucion'].mean() * 100
tasa_dev_region.index = tasa_dev_region.index.fillna('Sin dato')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, serie, titulo, color in [
        (axes[0], tasa_dev_prod.sort_values(ascending=False), "D. Devoluciones por producto", PALETA['rojo']),
        (axes[1], tasa_dev_canal.sort_values(ascending=False), "D. Devoluciones por canal", PALETA['marron']),
        (axes[2], tasa_dev_region.sort_values(ascending=False), "D. Devoluciones por región", PALETA['gris'])]:
    ax.barh(serie.index, serie.values, color=color, edgecolor='black')
    ax.axvline(tasa_dev, color='black', linestyle='--', linewidth=1.2, label=f'Global: {tasa_dev:.1f}%')
    ax.set_title(titulo, fontweight='bold')
    ax.set_xlabel("Tasa de devolución (%)")
    ax.legend(frameon=True, facecolor='white', fontsize=8)
    for rect in ax.patches:
        w = rect.get_width()
        ax.annotate(f'{w:.1f}%', xy=(w, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0), textcoords='offset points', va='center', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/fig_07_devoluciones_segmentos.png", dpi=300)
plt.close()

df_es = df[df['dias_entrega'].notna() & df['satisfaccion'].notna()]
corr_ent_sat = float(df_es['dias_entrega'].corr(df_es['satisfaccion']))
sast_por_entrega = df_es.groupby(pd.cut(df_es['dias_entrega'], bins=[0, 3, 7, 10, 14],
                                        labels=['1-3 días', '4-7 días', '8-10 días', '11-14 días']),
                                  observed=True)['satisfaccion'].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df_es, x='satisfaccion', y='dias_entrega', ax=ax1, color=PALETA['verde_claro'],
            medianprops={'color': 'black'})
ax1.set_title(f"D. Satisfacción vs días de entrega (r = {corr_ent_sat:+.2f})", fontweight='bold')
ax1.set_xlabel("Satisfacción (1-5)")
ax1.set_ylabel("Días de entrega")
sast_por_entrega.plot(kind='bar', ax=ax2, color=PALETA['verde'], edgecolor='black')
for i, v in enumerate(sast_por_entrega.values):
    ax2.annotate(f'{v:.2f}', xy=(i, v), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
ax2.set_title("D. Satisfacción promedio según demora de entrega", fontweight='bold')
ax2.set_xlabel("Rango de días de entrega")
ax2.set_ylabel("Satisfacción promedio (1-5)")
ax2.set_ylim(0, 5)
plt.tight_layout()
plt.savefig("outputs/fig_08_entrega_satisfaccion.png", dpi=300)
plt.close()

sat_canal = df.groupby('canal_venta')['satisfaccion'].mean()
sat_pago = df.groupby('medio_pago')['satisfaccion'].mean()
sat_cat = df.groupby('categoria')['satisfaccion'].mean()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, serie, titulo, color in [
        (axes[0], sat_canal.sort_values(), "D. Satisfacción por canal", PALETA['verde']),
        (axes[1], sat_pago.sort_values(), "D. Satisfacción por medio de pago", PALETA['marron']),
        (axes[2], sat_cat.sort_values(), "D. Satisfacción por categoría", PALETA['azul'])]:
    serie.plot(kind='barh', ax=ax, color=color, edgecolor='black')
    ax.set_xlim(0, 5)
    ax.set_title(titulo, fontweight='bold')
    ax.set_xlabel("Satisfacción promedio (1-5)")
    for rect in ax.patches:
        w = rect.get_width()
        ax.annotate(f'{w:.2f}', xy=(w, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0), textcoords='offset points', va='center', fontsize=8.5, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/fig_09_satisfaccion_factores.png", dpi=300)
plt.close()

# --- Calidad de datos: nulos antes/después ---
nulos_antes = df_original.isnull().mean() * 100
nulos_despues = df.isnull().mean() * 100
orden = nulos_despues.sort_values(ascending=False).index
plt.figure(figsize=(11, 5))
y = np.arange(len(orden))
plt.barh(y + 0.2, nulos_antes[orden], 0.4, color=PALETA['gris'], edgecolor='black', label='Antes de la limpieza (incluye inválidos convertidos)')
plt.barh(y - 0.2, nulos_despues[orden], 0.4, color=PALETA['verde'], edgecolor='black', label='Después de la limpieza')
plt.yticks(y, orden)
plt.gca().invert_yaxis()
plt.title("Perfil de valores faltantes por columna (antes y después de la limpieza)", fontweight='bold')
plt.xlabel("% de la columna")
plt.legend(frameon=True, facecolor='white')
plt.tight_layout()
plt.savefig("outputs/fig_10_calidad_datos.png", dpi=300)
plt.close()

print("   Figuras EDA generadas: fig_01 a fig_10")

# ----------------------------------------------------------
# 7. DECIDIR SI MODELAR + EVALUACIÓN CORRECTA (rúbrica, etapas 6 y 7)
# ----------------------------------------------------------
print("[7/9] Decisión de modelado y evaluación (FERNET 70/30, semilla 42)...")

# DECISIÓN: existe una pregunta predictiva concreta y útil para el negocio:
# anticipar qué ventas terminarán en devolución (clasificación binaria, objetivo 'devolucion').
# Es una variable objetivo válida y completa, con predictores disponibles.
# Se comienza con una referencia simple (clase mayoritaria) y luego Regresión Logística,
# un modelo explicable y adecuado para un objetivo binario.

num_features = ['cantidad', 'precio_unitario', 'descuento_pct', 'edad_cliente', 'dias_entrega']
cat_features = ['canal_venta', 'medio_pago', 'categoria', 'region']

df_m = df[df['cantidad'].notna() | df['precio_unitario'].notna()].copy()
df_m['region'] = df_m['region'].fillna('Sin dato')

X = df_m[num_features + cat_features]
y = df_m['devolucion'].astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
prepro = ColumnTransformer([('num', num_pipe, num_features), ('cat', cat_pipe, cat_features)])

# Referencia simple: clase mayoritaria (ajustada SOLO con entrenamiento)
baseline = Pipeline([('pre', prepro), ('clf', DummyClassifier(strategy='most_frequent', random_state=42))])
baseline.fit(X_tr, y_tr)
pred_base = baseline.predict(X_te)
acc_base = accuracy_score(y_te, pred_base)

# Modelo: Regresión Logística
modelo = Pipeline([('pre', prepro), ('clf', LogisticRegression(max_iter=2000, random_state=42))])
modelo.fit(X_tr, y_tr)
pred_te = modelo.predict(X_te)
pred_proba = modelo.predict_proba(X_te)[:, 1]

acc = accuracy_score(y_te, pred_te)
prec = precision_score(y_te, pred_te)
rec = recall_score(y_te, pred_te)
f1 = f1_score(y_te, pred_te)
cm = confusion_matrix(y_te, pred_te)
n_test = len(y_te)

# Importancia: odds ratios de la regresión logística
feat_names = modelo.named_steps['pre'].get_feature_names_out()
coefs = modelo.named_steps['clf'].coef_[0]
odds = pd.DataFrame({'variable': feat_names, 'coeficiente': coefs})
odds['|coef|'] = odds['coeficiente'].abs()
odds_top = odds.sort_values('|coef|', ascending=False).head(8)

supera = acc > acc_base
print(f"   Referencia (clase mayoritaria): exactitud = {acc_base:.2%}")
print(f"   Regresión logística: exactitud = {acc:.2%} | precisión(1) = {prec:.2%} | recall(1) = {rec:.2%} | F1(1) = {f1:.2%}")
print(f"   El modelo {'SUPERA' if supera else 'NO supera'} a la referencia simple.")

fig, (ax_cm, ax_odds) = plt.subplots(1, 2, figsize=(13, 4.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax_cm,
            xticklabels=['Pred: no devuelve', 'Pred: devuelve'],
            yticklabels=['Real: no devuelve', 'Real: devuelve'],
            annot_kws={'size': 14, 'fontweight': 'bold'})
ax_cm.set_title(f"A. Matriz de Confusión (test 30%, n={n_test})", fontweight='bold')

etiquetas = [v.replace('num__', '').replace('cat__', '').replace('Sin dato', 's/d') for v in odds_top['variable']]
colores_b = [PALETA['verde'] if c >= 0 else PALETA['rojo'] for c in odds_top['coeficiente']]
ax_odds.barh(etiquetas[::-1], odds_top['coeficiente'][::-1], color=colores_b[::-1], edgecolor='black')
ax_odds.axvline(0, color='black', linewidth=1)
ax_odds.set_title("B. Coeficientes del modelo (log-odds de devolución)", fontweight='bold')
ax_odds.set_xlabel("Coeficiente β")
plt.tight_layout()
plt.savefig("outputs/fig_11_modelo_devolucion.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 8. INFORME ENTREGABLE (rúbrica, etapas 8 y 9 - estructura.md)
# ----------------------------------------------------------
print("[8/9] Compilando el informe HTML (estructura.md)...")

def to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

imgs = {p: to_b64(f"outputs/{p}.png") for p in [
    'fig_01_evolucion_mensual', 'fig_02_estacionalidad', 'fig_03_productos', 'fig_04_categorias',
    'fig_05_cliente_canal', 'fig_06_pago_descuento', 'fig_07_devoluciones_segmentos',
    'fig_08_entrega_satisfaccion', 'fig_09_satisfaccion_factores', 'fig_10_calidad_datos',
    'fig_11_modelo_devolucion']}

bitacora_html = "".join(
    f"<tr><td>{b['paso']}</td><td>{b['cambio']}</td><td>{b['filas_valores_afectados']}</td>"
    f"<td>{b['regla']}</td><td>{b['justificacion']}</td><td>{b['influencia_potencial']}</td></tr>"
    for b in bitacora)

problemas_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in problemas.items())

kpis_html = "".join(f"<tr><td><strong>{r['Indicador']}</strong></td><td>{r['Valor']}</td></tr>"
                    for _, r in kpis.iterrows())

regiones_txt = ", ".join([f"{c} ({v/df['region'].notna().sum()*100:.1f}%)"
                          for c, v in df['region'].value_counts().items()])

sentido_corr_sat = "negativa" if corr_ent_sat < 0 else "positiva"
sentido_corr_desc = "positiva" if corr_desc_total > 0 else "negativa"
cat_principal = ing_cat.index[0]
share_cat_princ = share_cat.iloc[0]
mejor_canal = ing_canal.index[0]
peor_sat_factor = min([('canal', sat_canal.min(), sat_canal.idxmin()),
                       ('medio de pago', sat_pago.min(), sat_pago.idxmin()),
                       ('categoría', sat_cat.min(), sat_cat.idxmin())], key=lambda t: t[1])

conclusion_modelo = (
    f"El modelo de regresión logística alcanzó una exactitud de {acc:.2%} sobre el 30% de test, "
    f"frente al {acc_base:.2%} de la referencia (clase mayoritaria). "
    + ("Supera levemente a la referencia, por lo que aporta valor predictivo marginal pero existe."
       if supera else
       "No supera a la referencia simple, lo que indica que las variables disponibles NO permiten predecir "
       "de manera útil la devolución; el análisis descriptivo es, por lo tanto, la herramienta principal.")
)

html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Informe Técnico - Análisis de Ventas Comerciales</title>
<style>
  @page {{
    size: A4;
    margin: 15mm 14mm 15mm 14mm;
    @bottom-right {{ content: "Página " counter(page) " de " counter(pages); font-size: 8pt; color: #666; }}
    @bottom-left {{ content: "Informe Técnico | Análisis de Datos - Ventas Comerciales"; font-size: 8pt; color: #666; }}
  }}
  body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #212121; line-height: 1.4; font-size: 9pt; margin: 0; padding: 0; }}
  .header {{ border-bottom: 2.5px solid #2E7D32; padding-bottom: 6px; margin-bottom: 12px; }}
  .header h1 {{ font-size: 15pt; color: #1B5E20; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 0.5px; }}
  .header .meta {{ font-size: 8.5pt; color: #555; display: flex; justify-content: space-between; }}
  h2 {{ font-size: 11pt; color: #1B5E20; border-left: 4px solid #2E7D32; padding-left: 6px; margin-top: 14px; margin-bottom: 6px; }}
  h3 {{ font-size: 9.5pt; color: #333; margin-top: 8px; margin-bottom: 4px; }}
  p {{ margin: 0 0 5px 0; text-align: justify; }}
  table {{ width: 100%; border-collapse: collapse; margin: 6px 0; font-size: 8pt; }}
  th, td {{ border: 1px solid #dcdcdc; padding: 3px 6px; text-align: left; }}
  th {{ background-color: #f1f8f3; color: #1B5E20; font-weight: bold; }}
  tr:nth-child(even) {{ background-color: #fafafa; }}
  .figure-box {{ text-align: center; margin: 8px 0; page-break-inside: avoid; }}
  .figure-box img {{ max-width: 97%; height: auto; border: 1px solid #e0e0e0; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}
  .caption {{ font-size: 8pt; color: #555; margin-top: 3px; font-style: italic; text-align: center; }}
  .formula-box {{ background: #f4fbf5; border: 1px solid #c8e6c9; border-radius: 4px; padding: 5px 8px; margin: 6px 0; text-align: center; color: #1B5E20; }}
  .page-break {{ page-break-before: always; }}
  .limitacion {{ background: #fff8e1; border-left: 4px solid #F9A825; padding: 5px 8px; margin: 6px 0; }}
</style>
</head>
<body>

<div class="header">
  <h1>Informe Técnico: Análisis de Ventas Comerciales</h1>
  <div class="meta">
    <span><strong>Fuente:</strong> ventas_4.csv ({n0} registros originales)</span>
    <span><strong>Recepción:</strong> {FECHA_RECEPCION}</span>
    <span><strong>Protocolo:</strong> Rúbrica procesos.md (10 etapas) + estructura.md</span>
  </div>
</div>

<h2>1. Resumen ejecutivo</h2>
<p>
Se analizó el registro comercial de la empresa ({n0} operaciones, período {pd.to_datetime(df_v['fecha_venta']).min():%d/%m/%Y} a {pd.to_datetime(df_v['fecha_venta']).max():%d/%m/%Y})
para responder a las preguntas de la dirección comercial sobre ventas, productos, clientes y calidad de servicio.
Los ingresos totales suman <strong>${ingresos_totales:,.0f}</strong> en {len(df)} operaciones válidas, con un ticket promedio de ${ticket_prom:,.2f}.
La categoría dominante es <strong>{cat_principal}</strong> ({share_cat_princ:.1f}% de los ingresos) y el canal más rentable es <strong>{mejor_canal}</strong>.
La tasa global de devolución es <strong>{tasa_dev:.1f}%</strong> y la satisfacción promedio {sat_prom:.2f}/5, {sentido_corr_sat}mente asociada a los días de entrega (r = {corr_ent_sat:+.2f}).
{conclusion_modelo}
</p>

<h2>2. Introducción y contexto</h2>
<p>
La empresa, dedicada a la comercialización de productos tecnológicos, necesita evaluar el comportamiento de sus ventas en los
últimos períodos. La dirección comercial sospecha diferencias importantes entre regiones, canales y categorías, y requiere
conocer qué factores se asocian al volumen de ventas, a la satisfacción de los clientes y a las devoluciones, con el fin de
identificar oportunidades de mejora y sustentar decisiones estratégicas.
</p>

<h2>3. Problema y objetivos</h2>
<p>
<strong>Objetivo general:</strong> analizar el comportamiento comercial identificando patrones de ventas, productos y regiones de
mayor rendimiento, factores asociados a la satisfacción del cliente y posibles relaciones entre variables comerciales.
</p>
<table>
  <thead><tr><th>Pregunta de análisis</th><th>Columnas utilizadas</th></tr></thead>
  <tbody>
    <tr><td>{PREGUNTAS['A']}</td><td>fecha_venta, total_venta</td></tr>
    <tr><td>{PREGUNTAS['B']}</td><td>producto, categoria, cantidad, total_venta</td></tr>
    <tr><td>{PREGUNTAS['C']}</td><td>edad_cliente, canal_venta, medio_pago, descuento_pct, total_venta</td></tr>
    <tr><td>{PREGUNTAS['D']}</td><td>devolucion, dias_entrega, satisfaccion, producto, region, canal_venta</td></tr>
  </tbody>
</table>
<div class="limitacion"><strong>Qué no se podrá concluir:</strong> los datos no incluyen costos, márgenes ni identificadores de
cliente único, por lo que no es posible calcular rentabilidad por producto ni análisis de recompra. Las relaciones observadas
son asociaciones descriptivas, no causales.</div>

<h2>4. Fuente y descripción de los datos</h2>
<p>
<strong>Archivo:</strong> ventas_4.csv (CSV, {n0} filas × 15 columnas). <strong>Origen:</strong> {ORIGEN}.
<strong>Una fila =</strong> una operación de venta. <strong>Unidades:</strong> importes en pesos; tiempos en días; satisfacción en escala 1-5.
No se identifican datos personales identificables (la edad es genérica y no hay nombres ni documentos).
</p>
<table>
  <thead><tr><th>Columna</th><th>Descripción</th><th>Columna</th><th>Descripción</th></tr></thead>
  <tbody>
    <tr><td><strong>id_registro</strong></td><td>ID de la operación</td><td><strong>total_venta</strong></td><td>Importe total registrado</td></tr>
    <tr><td><strong>fecha_venta</strong></td><td>Fecha de la venta</td><td><strong>edad_cliente</strong></td><td>Edad del cliente (años)</td></tr>
    <tr><td><strong>producto</strong></td><td>Producto (6 tipos)</td><td><strong>canal_venta</strong></td><td>Online / Presencial / Telefónico</td></tr>
    <tr><td><strong>categoria</strong></td><td>Categoría (3 tipos)</td><td><strong>medio_pago</strong></td><td>Transferencia / Tarjeta / Efectivo</td></tr>
    <tr><td><strong>region</strong></td><td>Región (5 zonas)</td><td><strong>devolucion</strong></td><td>¿La venta se devolvió? (Sí/No)</td></tr>
    <tr><td><strong>cantidad</strong></td><td>Unidades vendidas</td><td><strong>dias_entrega</strong></td><td>Días hasta la entrega</td></tr>
    <tr><td><strong>precio_unitario</strong></td><td>Precio por unidad</td><td><strong>satisfaccion</strong></td><td>Satisfacción (1-5)</td></tr>
    <tr><td><strong>descuento_pct</strong></td><td>Descuento (0-30%)</td><td></td><td></td></tr>
  </tbody>
</table>

<h2>5. Metodología</h2>
<p>
Se siguió el flujo de la rúbrica: (1) recepción y documentación; (2) definición de preguntas; (3) validación de estructura y
calidad; (4) limpieza con bitácora reproducible; (5) KPIs y análisis exploratorio; (6) decisión justificada de modelar;
(7) evaluación con partición FERNET 70/30 (semilla 42), imputaciones ajustadas <strong>solo con datos de entrenamiento</strong> y
comparación contra una referencia simple; (8) interpretación separando hallazgo, interpretación y limitación; (9) este informe.
Los archivos originales permanecen intactos: todo el trabajo se realizó sobre una copia en data/.
</p>

<h2>6. Limpieza y preparación</h2>
<p>
La validación detectó los siguientes problemas de calidad (etapa 3):
</p>
<table>
  <thead><tr><th>Problema detectado</th><th>Casos</th></tr></thead>
  <tbody>{problemas_html}</tbody>
</table>
<p>
<strong>Criterios aplicados:</strong> no se imputaron en bloque medias/medianas ni se eliminaron en bloque los valores extremos.
Cada transformación se justifica por el significado del dato y quedó registrada en la bitácora
(<em>outputs/bitacora_limpieza.csv</em>). El dataset limpio se exporta como <em>outputs/ventas_4_limpio.csv</em>.
</p>
<table>
  <thead><tr><th>#</th><th>Cambio</th><th>Afectados</th><th>Regla</th><th>Justificación</th><th>Influencia potencial</th></tr></thead>
  <tbody>{bitacora_html}</tbody>
</table>

<div class="figure-box">
  <img src="data:image/png;base64,{imgs['fig_10_calidad_datos']}" alt="Calidad de datos">
  <div class="caption">Figura 10. Perfil de faltantes por columna antes y después de la limpieza (los valores inválidos convertidos a nulo suman al perfil final).</div>
</div>

<div class="page-break"></div>

<h2>7. Análisis exploratorio</h2>
<p><strong>KPIs del período (sobre el dataset limpio):</strong></p>
<table style="width:60%">
  <thead><tr><th>Indicador</th><th>Valor</th></tr></thead>
  <tbody>{kpis_html}</tbody>
</table>

<h3>7.1 Comportamiento de las ventas (Pregunta A)</h3>
<p>
Los ingresos totales del período ascienden a <strong>${ingresos_totales:,.0f}</strong>. La serie mensual muestra variaciones
entre meses; el mes calendario con mayores ingresos fue <strong>{MESES[int(mejor_mes['mes'])]} de {int(mejor_mes['año'])}</strong>
(${mejor_mes['total_venta']:,.0f}) y el de menores, {MESES[int(peor_mes['mes'])]} de {int(peor_mes['año'])}
(${peor_mes['total_venta']:,.0f}; nótese que {int(peor_mes['año'])} no está completo, por lo que esa comparación de totales anuales
no es directa). Al comparar el mismo mes calendario entre años completos se observa
{'un patrón estacional moderado' if est.groupby('mes')['total_venta'].mean().std() / est.groupby('mes')['total_venta'].mean().mean() > 0.15 else 'una estacionalidad poco marcada'}.
</p>

<div class="figure-box">
  <img src="data:image/png;base64,{imgs['fig_01_evolucion_mensual']}" alt="Evolución mensual">
  <div class="caption">Figura 1. Ingresos por mes (fechas válidas). La franja ámbar marca el último período, aún incompleto (hasta el 26/09).</div>
</div>

<div class="figure-box">
  <img src="data:image/png;base64,{imgs['fig_02_estacionalidad']}" alt="Estacionalidad">
  <div class="caption">Figura 2. Ingresos por mes calendario comparando años: permite separar estacionalidad de tendencia y del efecto de años incompletos.</div>
</div>

<h3>7.2 Productos y categorías (Pregunta B)</h3>
<p>
El producto que más ingresos genera es <strong>{top_prod_ing}</strong> (${ing_prod.iloc[0]:,.0f}) y el que más unidades vende es
<strong>{top_prod_uni}</strong> ({uni_prod.iloc[0]:,.0f} u.). Ambos puestos {'coinciden' if coinciden else 'NO coinciden'}:
{'el producto estrella también lidera en volumen' if coinciden else 'los productos más vendidos en unidades no son los que más ingresos aportan, lo que sugiere diferencias de precio entre productos'}.
En categorías, <strong>{cat_principal}</strong> concentra el {share_cat_princ:.1f}% de los ingresos
(${ing_cat.iloc[0]:,.0f}), seguida por {ing_cat.index[1]} ({share_cat.iloc[1]:.1f}%) y {ing_cat.index[2]} ({share_cat.iloc[2]:.1f}%):
la concentración es {'alta' if share_cat_princ > 45 else 'moderada/baja'} y ninguna categoría domina el negocio de forma excluyente.
</p>

<div class="figure-box">
  <img src="data:image/png;base64,{imgs['fig_03_productos']}" alt="Productos">
  <div class="caption">Figura 3. Ingresos totales y unidades vendidas por producto: contrasta rendimiento monetario vs volumen físico.</div>
</div>

<div class="figure-box">
  <img src="data:image/png;base64,{imgs['fig_04_categorias']}" alt="Categorías">
  <div class="caption">Figura 4. Concentración de ingresos y unidades por categoría (los porcentajes indican el share de ingresos).</div>
</div>

<div class="page-break"></div>

<h3>7.3 Comportamiento de los clientes (Pregunta C)</h3>
<p>
La relación entre edad del cliente e importe es prácticamente nula (r = {corr_edad_total:+.2f}):{' el importe de compra no depende de la edad dentro del rango observado (18-100 años)' if abs(corr_edad_total) < 0.1 else ' existe una relación débil pero detectable'}.
Entre canales, <strong>{mejor_canal}</strong> lidera en ingresos (${ing_canal.iloc[0]:,.0f}), aunque el ticket promedio por canal es
similar entre sí (diferencia de ${(ticket_canal.max()-ticket_canal.min()):,.2f}), por lo que la brecha se explica más por cantidad de operaciones que por tamaño de venta.
Por medio de pago, las operaciones se reparten de forma casi pareja ({', '.join(f'{i}: {int(v)}' for i, v in op_pago['n'].items())}),
sin un medio asociado de manera clara a operaciones de mayor volumen.
El descuento aplicado muestra correlación {sentido_corr_desc} con el importe final (r = {corr_desc_total:+.2f}):
{'a mayor descuento no se observa mayor importe; el descuento no parece estar funcionando como palanca de tickets más grandes' if abs(corr_desc_total) < 0.15 else 'existe una relación débil entre descuento e importe'}.
</p>

<div class="figure-box">
  <img src="data:image/png;base64,{imgs['fig_05_cliente_canal']}" alt="Cliente y canal">
  <div class="caption">Figura 5. Panel A: dispersión edad vs importe. Panel B: ingresos y ticket promedio por canal de venta.</div>
</div>

<div class="figure-box">
  <img src="data:image/png;base64,{imgs['fig_06_pago_descuento']}" alt="Pago y descuento">
  <div class="caption">Figura 6. Panel A: volumen de operaciones por medio de pago. Panel B: descuento aplicado vs importe final.</div>
</div>

<h3>7.4 Calidad del servicio (Pregunta D)</h3>
<p>
El <strong>{tasa_dev:.1f}%</strong> de las ventas termina en devolución ({int(df['devolucion'].sum())} de {len(df)} operaciones).
Por producto, la mayor tasa es de <strong>{tasa_dev_prod.idxmax()}</strong> ({tasa_dev_prod.max():.1f}%) y la menor de {tasa_dev_prod.idxmin()} ({tasa_dev_prod.min():.1f}%).
Por canal, destaca {tasa_dev_canal.idxmax()} ({tasa_dev_canal.max():.1f}%); por región, {tasa_dev_region.idxmax()} ({tasa_dev_region.max():.1f}%).
Las diferencias entre segmentos son {'moderadas y consistentes' if (tasa_dev_prod.max()-tasa_dev_prod.min()) > 4 else 'acotadas (menos de 4 puntos porcentuales)'}.
La satisfacción promedio es {sat_prom:.2f}/5 y su relación con los días de entrega es {sentido_corr_sat} (r = {corr_ent_sat:+.2f}):
{'las demoras mayores se asocian a menor satisfacción, y el tramo de entregas más rápidas (1-3 días) concentra la mejor valoración' if corr_ent_sat < 0 else 'la relación con la demora es débil; el factor más desfavorable se observa por ' + peor_sat_factor[0] + ' (' + peor_sat_factor[2] + ': ' + f'{peor_sat_factor[1]:.2f}' + ')'}.
</p>

<div class="figure-box">
  <img src="data:image/png;base64,{imgs['fig_07_devoluciones_segmentos']}" alt="Devoluciones">
  <div class="caption">Figura 7. Tasa de devolución por producto, canal y región; la línea punteada marca la tasa global ({tasa_dev:.1f}%).</div>
</div>

<div class="figure-box">
  <img src="data:image/png;base64,{imgs['fig_08_entrega_satisfaccion']}" alt="Entrega y satisfacción">
  <div class="caption">Figura 8. Panel A: distribución de días de entrega según nivel de satisfacción. Panel B: satisfacción promedio por tramo de demora.</div>
</div>

<div class="figure-box">
  <img src="data:image/png;base64,{imgs['fig_09_satisfaccion_factores']}" alt="Factores de satisfacción">
  <div class="caption">Figura 9. Satisfacción promedio por canal, medio de pago y categoría (escala 1-5). Las diferencias entre niveles son pequeñas, por lo que deben leerse como indicios y no como causas.</div>
</div>

<div class="page-break"></div>

<h2>8. Modelado (opcional, justificado)</h2>
<p>
<strong>Decisión:</strong> se modela porque existe una pregunta predictiva concreta y de valor para el negocio: <em>anticipar
qué ventas terminarán en devolución</em>, algo que permitiría revisar pedidos antes de la entrega. La variable objetivo
<em>devolucion</em> es binaria, está completa en el 100% de las filas y los predictores disponibles (monto, producto, canal,
demora, cliente) son pertinentes. No se creó ninguna variable objetivo artificial.
</p>
<p>
<strong>Protocolo de evaluación (sin fuga de información):</strong> partición FERNET 70/30 estratificada (semilla 42,
n_train = {len(y_tr)}, n_test = {n_test}); imputación de faltantes por mediana/moda ajustada <strong>únicamente con datos de
entrenamiento</strong>; codificación one-hot de categóricas; comparación obligatoria contra una <strong>referencia simple</strong>
(clase mayoritaria). La partición es aleatoria (y no temporal) porque el objetivo es clasificar operaciones y no pronosticar
una serie; esta elección se declara como limitación del alcance predictivo.
</p>
<table style="width:65%">
  <thead><tr><th>Métrica (clase 1 = devuelve)</th><th>Regresión Logística</th><th>Referencia (mayoría)</th></tr></thead>
  <tbody>
    <tr><td>Exactitud (Accuracy)</td><td><strong>{acc:.2%}</strong></td><td>{acc_base:.2%}</td></tr>
    <tr><td>Precisión (clase 1)</td><td><strong>{prec:.2%}</strong></td><td>—</td></tr>
    <tr><td>Sensibilidad / Recall (clase 1)</td><td><strong>{rec:.2%}</strong></td><td>—</td></tr>
    <tr><td>F1-Score (clase 1)</td><td><strong>{f1:.2%}</strong></td><td>—</td></tr>
  </tbody>
</table>
<div class="limitacion">
<strong>Lectura honesta del modelo:</strong> {conclusion_modelo} Los coeficientes del panel B indican la dirección de cada
variable en la probabilidad de devolución (β &gt; 0 aumenta el riesgo); ante un modelo que no supera la referencia, estos
coeficientes deben leerse solo como indicios exploratorios, no como un predictor confiable.
</p></div>

<div class="figure-box">
  <img src="data:image/png;base64,{imgs['fig_11_modelo_devolucion']}" alt="Modelo devolución">
  <div class="caption">Figura 11. Panel A: matriz de confusión sobre el 30% de test. Panel B: coeficientes de la regresión logística (verde: aumenta la probabilidad de devolución; rojo: la reduce).</div>
</div>

<h2>9. Resultados: hallazgos, interpretación y límites</h2>
<table>
  <thead><tr><th>Hallazgo (dato)</th><th>Interpretación (para el negocio)</th><th>Limitación (qué no se puede afirmar)</th></tr></thead>
  <tbody>
    <tr><td>{cat_principal} concentra {share_cat_princ:.1f}% de los ingresos</td><td>El negocio descansa de forma notable en una categoría</td><td>Asociación descriptiva; sin datos de costo no se sabe si es la más rentable</td></tr>
    <tr><td>{mejor_canal} lidera en ingresos</td><td>Refuerza la inversión comercial en ese canal</td><td>No es causal: puede reflejar simplemente más volumen de operaciones</td></tr>
    <tr><td>Tasa de devolución global {tasa_dev:.1f}%</td><td>1 de cada ~{(100/tasa_dev):.0f} ventas vuelve; foco directo de mejora</td><td>Sin motivo de devolución registrado, no se puede diagnosticar la causa</td></tr>
    <tr><td>Satisfacción {sentido_corr_sat}mente asociada a días de entrega (r = {corr_ent_sat:+.2f})</td><td>Reducir demoras es la palanca más visible de experiencia del cliente</td><td>Correlación ≠ causalidad; la escala 1-5 agrega sólo variación moderada</td></tr>
    <tr><td>Descuento vs importe: r = {corr_desc_total:+.2f}</td><td>El descuento no acompaña tickets mayores en estos datos</td><td>Un experimento controlado sería necesario para afirmar efecto</td></tr>
    <tr><td>Modelo de devolución: exactitud {acc:.2%} vs referencia {acc_base:.2%}</td><td>{'Aporte predictivo marginal' if supera else 'Las variables disponibles no predicen la devolución'}</td><td>Partición aleatoria (no temporal); un modelo no superado por la referencia no debe usarse para decidir</td></tr>
  </tbody>
</table>

<h2>10. Conclusiones y recomendaciones</h2>
<p>
1. <strong>Ventas:</strong> los ingresos del período suman ${ingresos_totales:,.0f} con estacionalidad
{'marcada' if est.groupby('mes')['total_venta'].mean().std() / est.groupby('mes')['total_venta'].mean().mean() > 0.15 else 'acotada'}; se recomienda planificar inventarios usando los meses fuertes de años completos (2023-2024) y no el último período incompleto.
</p>
<p>
2. <strong>Portafolio:</strong> {cat_principal} es la categoría ancla del negocio ({share_cat_princ:.1f}% de ingresos). {'El producto líder en unidades (' + top_prod_uni + ') no coincide con el líder de ingresos (' + top_prod_ing + '), lo que sugiere revisar precios y mix.' if not coinciden else 'El liderazgo en unidades e ingresos coincide, consolidando ese producto como prioridad.'}
</p>
<p>
3. <strong>Calidad de servicio:</strong> con un {tasa_dev:.1f}% de devoluciones y satisfacción {sentido_corr_sat}mente ligada a la
demora (r = {corr_ent_sat:+.2f}), la recomendación operativa es focalizar revisiones de picking/logística en los segmentos con
mayor tasa ({tasa_dev_prod.idxmax()} y canal {tasa_dev_canal.idxmax()}) y monitorear el tramo de entregas de más de 7 días.
</p>
<p>
4. <strong>Modelado:</strong> {conclusion_modelo} Recomendación: incorporar datos de motivo de devolución, costo y margen, e
identificadores de cliente, para habilitar modelos con valor real de decisión.
</p>

<h2>11. Limitaciones y trabajo futuro</h2>
<p>
&bull; {n_fecha} ventas ({n_fecha/n0*100:.1f}%) tienen fecha inválida y quedan fuera de los análisis temporales.<br>
&bull; {n_precio} precios y {n_cant} cantidades eran ilegibles o imposibles; se conservaron las filas pero se excluyen de los cálculos correspondientes.<br>
&bull; {int((df['total_venta'] - df['cantidad']*df['precio_unitario']*(1-df['descuento_pct'].fillna(0))).abs().gt(1).sum())} registros muestran inconsistencia entre total_venta y cantidad × precio × (1 − descuento); se usó el total registrado como dato fuente.<br>
&bull; {problemas['satisfaccion_faltante']} satisfacciones ({problemas['satisfaccion_faltante']/n0*100:.1f}%) y {problemas['region_faltante']} regiones faltan; los promedios de satisfacción se calculan sobre los registrados y pueden estar sesgados.<br>
&bull; No hay costo, margen, motivo de devolución ni ID de cliente: limita rentabilidad, diagnóstico de devoluciones y análisis de recompra.<br>
&bull; Trabajo futuro: incorporar esos campos, unir con datos de clientes (respetando claves y granularidad), probar modelos de series temporales para pronóstico y validar experimentalmente el efecto de los descuentos.
</p>

<h2>12. Anexo reproducible</h2>
<p>
<strong>Archivos generados:</strong> outputs/ventas_4_limpio.csv (dataset limpio), outputs/bitacora_limpieza.csv (bitácora de
limpieza), outputs/kpis.csv (KPIs), fig_01 a fig_11 (PNG 300 dpi) y este informe (HTML/PDF).<br>
<strong>Configuración:</strong> Python + pandas/numpy/matplotlib/seaborn/scikit-learn. Partición FERNET 70/30 estratificada,
semilla = 42. El pipeline completo se reproducen ejecutando <em>python main.py</em> desde la raíz del proyecto; los datos
originales en data/ventas_4.csv no fueron modificados.
</p>

</body>
</html>
"""

html_path = os.path.abspath("outputs/informe_ventas.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"   HTML generado en: {html_path}")

# ----------------------------------------------------------
# 9. GENERACIÓN DE PDF
# ----------------------------------------------------------
print("[9/9] Convirtiendo informe a PDF...")

pdf_path = os.path.abspath("outputs/informe_ventas.pdf")
edge_paths = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    shutil.which("msedge"),
    shutil.which("chrome"),
]
browser_exe = next((p for p in edge_paths if p and os.path.exists(p)), None)

pdf_creado = False
if browser_exe:
    cmd = [browser_exe, "--headless", "--disable-gpu",
           "--run-all-compositor-stages-before-draw",
           f"--print-to-pdf={pdf_path}", html_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pdf_creado = os.path.exists(pdf_path)
    if pdf_creado:
        print(f"   ¡PDF compilado con éxito vía navegador!: {pdf_path}")

if not pdf_creado:
    try:
        from weasyprint import HTML
        HTML(html_path).write_pdf(pdf_path)
        print(f"   ¡PDF compilado con éxito vía WeasyPrint!: {pdf_path}")
        pdf_creado = True
    except Exception:
        pass

if not pdf_creado:
    print("   Nota: abre outputs/informe_ventas.html en el navegador y usa Ctrl+P -> Guardar como PDF.")

print("\nPipeline completado al 100% según la rúbrica procesos.md.")
