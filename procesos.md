Proceso de análisis de datos

Propósito

Este documento define el flujo de trabajo para analizar conjuntos de datos y
producir un informe técnico entregable. El análisis debe adaptarse a cada
conjunto: no se aplican siempre las mismas transformaciones, métricas ni
modelos.

La regla central es partir de una pregunta y de los datos disponibles, no de
un algoritmo. El modelado es opcional y solo se realiza si responde una
necesidad concreta.

Principios de trabajo





Conservar los archivos originales sin modificarlos. Trabajar sobre copias.



Entender qué representa cada fila, columna, unidad, período y población.



Relacionar cada análisis y conclusión con una pregunta explícita.



Justificar las decisiones de limpieza y modelado; no aplicar recetas a ciegas.



Evitar fuga de información entre entrenamiento y evaluación.



Distinguir descripción, asociación, predicción y causalidad.



Informar limitaciones y preguntas que los datos no permiten responder.



Mantener registro suficiente para reproducir el análisis.

Flujo de trabajo

1. Recibir y documentar los datos

Al recibir los archivos:





Registrar nombre, formato, origen conocido y fecha de recepción.



Identificar cuántos archivos hay y si se relacionan entre sí.



Mantener intacta una copia original.



Registrar el significado conocido de columnas, unidades, períodos y población.



Identificar posibles datos personales o sensibles y tratarlos con cuidado.

Preguntas:





¿Quién produjo los datos y con qué propósito?



¿Qué representa cada fila y cada columna?



¿Qué población, lugar y período describen?



¿En qué unidades se expresan las variables?



¿Hay información sensible que deba protegerse?

Resultado: inventario de archivos y descripción inicial del conjunto.

2. Definir el problema y las preguntas

Redactar el propósito en lenguaje claro antes de elegir técnicas. Determinar si
se busca:





Describir qué ocurrió.



Comparar grupos o períodos.



Explorar relaciones entre variables.



Predecir un valor numérico.



Clasificar casos en categorías.



Analizar la evolución temporal.



Encontrar grupos de observaciones similares.

Para cada pregunta, documentar:





La pregunta concreta.



Las columnas necesarias.



La población y el período a los que aplica.



Qué resultado sería útil.



Qué no se podrá concluir.

Comprobar que los datos realmente contienen la información necesaria. Si falta
una variable, una fecha o contexto suficiente, reformular la pregunta o dejar
constancia de la limitación. No inventar respuestas mediante un modelo.

Resultado: problema, objetivos y preguntas de análisis acordadas.

3. Validar la estructura y la calidad

Inspeccionar cada archivo antes de transformarlo:





Cantidad de filas y columnas.



Nombres y tipos de columnas.



Valores faltantes y formatos inconsistentes.



Filas duplicadas.



Columnas constantes o con cardinalidad muy baja.



Posibles identificadores.



Fechas, unidades, categorías y rangos.



Valores sospechosos o fuera de los límites esperables.

Preguntas:





¿Cada fila representa una observación esperada?



¿Los tipos de datos son correctos?



¿Los rangos son plausibles para el contexto?



¿Los duplicados son errores o registros legítimos?



¿Hay columnas que identifiquen personas o entidades?

Resultado: perfil de calidad con problemas detectados y ejemplos.

4. Limpiar y preparar los datos

Aplicar únicamente transformaciones justificadas por el significado de los
datos. Algunas acciones posibles:





Normalizar espacios y variantes de escritura en categorías.



Convertir columnas a tipos numéricos o fechas.



Eliminar duplicados solo cuando representen registros repetidos.



Tratar faltantes según su cantidad, distribución y contexto.



Investigar valores atípicos antes de decidir si son errores o casos válidos.



Unir archivos solo después de comprobar claves y nivel de granularidad.

Por cada cambio registrar:





Qué se modificó.



Cuántas filas o valores afectó.



Qué regla se usó.



Por qué la regla tiene sentido.



Cómo podría influir en los resultados.

No reemplazar automáticamente todos los faltantes por una media o mediana ni
eliminar todos los valores extremos. Un valor inusual puede ser un error, pero
también una observación real e importante.

Resultado: dataset preparado y bitácora reproducible de limpieza.

5. Realizar análisis exploratorio

Seleccionar tablas y gráficos que ayuden a responder las preguntas, evitando
visualizaciones por rutina.

Según el tipo de variable, pueden resultar útiles:





Numéricas: distribución, mediana, rango, dispersión y valores extremos.



Categóricas: frecuencias, proporciones y categorías poco frecuentes.



Relaciones: dispersión, tablas cruzadas o correlaciones cuando sean
pertinentes.



Fechas: tendencias, estacionalidad, cambios y períodos incompletos.



Grupos o regiones: comparaciones con tamaños de muestra visibles.

Preguntas:





¿Qué patrones aparecen?



¿Hay diferencias entre grupos o períodos?



¿La muestra alcanza para sostener esa comparación?



¿El patrón puede depender de faltantes, valores extremos o sesgos?



¿El gráfico muestra unidades y denominadores correctamente?

Una asociación observada no demuestra causalidad.

Resultado: hallazgos descriptivos respaldados por tablas y gráficos.

6. Decidir si corresponde modelar

El modelado es opcional. Considerarlo cuando exista una pregunta predictiva o de
clasificación concreta, los datos sean pertinentes y el modelo pueda aportar
algo útil frente a un análisis descriptivo sencillo.







Necesidad



Enfoque posible





Predecir un valor numérico



Regresión





Asignar una clase a un caso



Clasificación





Estimar valores futuros en el tiempo



Serie temporal





Identificar grupos de casos similares



Segmentación / clustering





Describir y entender el conjunto



Análisis descriptivo, sin modelo predictivo

Antes de entrenar, confirmar el objetivo, los datos de entrada y el uso
esperado. La selección del algoritmo sucede después de entender la pregunta y
los datos. Conviene empezar con una referencia simple y explicable antes de
probar métodos más complejos.

No modelar si no hay una pregunta predictiva, una variable objetivo válida o
datos suficientes. No crear una variable objetivo artificial para incluir un
modelo en el informe.

Resultado: decisión documentada de modelar o no, con su justificación.

7. Preparar y evaluar modelos correctamente

Si se decide modelar:





Separar entrenamiento y evaluación de forma adecuada a la tarea.



En datos temporales, respetar el orden del tiempo; no mezclar pasado y futuro
al azar.



En datos agrupados, evitar que observaciones relacionadas queden
indebidamente en entrenamiento y prueba.



Ajustar imputaciones, escalado y selección de variables usando solo datos de
entrenamiento.



No filtrar ni transformar datos usando información del conjunto de prueba.



Comparar con una referencia sencilla y registrar partición y semilla.

Elegir métricas según el problema:





Regresión: MAE, RMSE, R² y, cuando corresponda, MSE.



Clasificación: matriz de confusión, precision, recall y F1. Usar métricas
adicionales cuando sean apropiadas y calculables.



Clustering: métricas internas como indicios, no como prueba de que los
grupos sean útiles.



Series temporales: evaluar con períodos futuros reservados y métricas
coherentes con la decisión que se tomará.

No evaluar un modelo con una única cifra. Explicar el significado de cada
métrica en la escala y contexto del problema, compararla con una referencia y
revisar los errores relevantes.

Resultado: modelo, protocolo de evaluación, métricas, diagnósticos y
explicación de errores.

8. Interpretar resultados y conclusiones

Para cada resultado importante, separar:





Hallazgo: qué muestran los datos o el modelo.



Interpretación: qué significa para la pregunta original.



Limitación: qué no se puede afirmar con estos datos o método.

Preguntas de control:





¿El análisis responde a la pregunta inicial?



¿La conclusión está respaldada por resultados observables?



¿Se está confundiendo correlación con causalidad?



¿Se generaliza más allá de la muestra, período o población?



¿Qué errores del modelo podrían tener consecuencias importantes?



¿Qué información adicional podría cambiar la conclusión?

Si una pregunta queda sin respuesta, decirlo claramente y explicar por qué.

9. Elaborar el informe entregable

Estructura recomendada:





Resumen ejecutivo: pregunta, enfoque y conclusión principal.



Contexto, problema y objetivos.



Fuente y descripción de los datos.



Preguntas de análisis.



Metodología: limpieza, exploración y, si corresponde, modelado.



Calidad y limitaciones de los datos.



Análisis exploratorio: tablas y visualizaciones pertinentes.



Modelado y evaluación, solo si corresponde.



Resultados: hallazgos, interpretación y límites.



Conclusiones y recomendaciones.



Limitaciones y trabajo futuro.



Anexo reproducible: decisiones, configuración y archivos procesados
 relevantes.

Cada gráfico debe tener título, unidades, leyenda cuando corresponda y una
explicación de su aporte. El informe no debe contener cifras, comparaciones ni
recomendaciones que no se hayan calculado o justificado.

10. Verificar antes de entregar





¿Los nombres de archivos y conteos coinciden con los datos procesados?



¿Los archivos originales quedaron intactos?



¿Todas las transformaciones están documentadas?



¿La evaluación evita fuga de información?



¿Las respuestas se conectan con las preguntas iniciales?



¿Las conclusiones distinguen asociación, predicción y causalidad?



¿Las tablas y gráficos son legibles?



¿Se puede volver a ejecutar el proceso?

Relación entre preguntas, análisis y conclusiones

Para mantener trazabilidad, cada pregunta debe tener una cadena explícita:

Pregunta
  → datos y columnas necesarios
  → método o visualización elegida
  → resultado obtenido
  → interpretación
  → limitaciones
  → conclusión que responde a la pregunta

Si algún eslabón falta, no se debe presentar una conclusión como respuesta
definitiva.

Resumen

El flujo completo es:

recibir y entender → formular preguntas → validar → limpiar con justificación
→ explorar → decidir si modelar → evaluar correctamente → interpretar
→ comunicar resultados y límites.

Lo que puede cambiar entre proyectos son las preguntas, transformaciones,
gráficos, modelos y métricas. Lo que debe mantenerse constante es la
trazabilidad, la justificación de las decisiones y la honestidad sobre lo que
los datos permiten concluir.