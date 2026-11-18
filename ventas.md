Dataset : Ventas comerciales
1. Contexto del caso
Una empresa dedicada a la comercialización de productos tecnológicos necesita evaluar el comportamiento de sus ventas durante los últimos períodos. La dirección comercial sospecha que existen diferencias importantes entre regiones, canales de comercialización y categorías de productos.
También necesita conocer qué factores pueden estar relacionados con el volumen de ventas, la satisfacción de los clientes y las devoluciones.
La empresa solicita un análisis de sus registros comerciales para identificar oportunidades de mejora y proporcionar información que permita tomar decisiones estratégicas.
2. Objetivo general
Analizar el comportamiento comercial de la empresa, identificando patrones de ventas, productos y regiones de mayor rendimiento, factores asociados a la satisfacción del cliente y posibles relaciones entre las variables comerciales.
3. Preguntas orientadoras
Propongo los siguientes cuestionamientos:
A. Comportamiento de las ventas
•	¿Cuál es el volumen total de ventas?
•	¿Cómo evolucionan las ventas a lo largo del tiempo?
•	¿Qué meses presentan mayores y menores ingresos?
•	¿Existe estacionalidad en el comportamiento comercial?
B. Productos y categorías
•	¿Qué productos generan mayores ingresos?
•	¿Qué productos presentan mayor cantidad de unidades vendidas?
•	¿Existe alguna categoría que concentre una proporción significativa de los ingresos?
•	¿Los productos más vendidos son también los que generan mayores ingresos?
C. Comportamiento de los clientes
•	¿Qué relación existe entre la edad de los clientes y sus compras?
•	¿Qué canal de venta presenta mejores resultados?
•	¿Existe relación entre el medio de pago y el volumen de las operaciones?
•	¿Qué relación existe entre los descuentos aplicados y el importe final de las ventas?
D. Calidad del servicio
•	¿Qué porcentaje de ventas termina en devoluciones?
•	¿Existe relación entre los días de entrega y la satisfacción del cliente?
•	¿Las devoluciones presentan algún patrón por producto, región o canal?
•	¿Qué factores parecen estar asociados a una menor satisfacción?
4. KPIs sugeridos
Indicador	Cálculo
Ingresos totales	Suma de total_venta
Unidades vendidas	Suma de cantidad
Precio promedio	Promedio de precio_unitario
Ticket promedio	Ingresos / cantidad de operaciones
Tasa de devolución	Ventas devueltas / ventas totales × 100
Descuento promedio	Promedio de descuento_pct
Satisfacción promedio	Promedio de satisfacción
Tiempo promedio de entrega	Promedio de dias_entrega

Modelado (opcional y sujeto a justificación):
A partir de los resultados del análisis exploratorio, evaluar la posibilidad de desarrollar uno o más modelos de Machine Learning. Los estudiantes deberán determinar si las variables disponibles permiten abordar un problema de predicción, clasificación o agrupamiento.
En caso de implementar modelos, deberán:
•	Justificar la técnica seleccionada.
•	Identificar las variables predictoras y la variable objetivo, cuando corresponda.
•	Realizar la preparación de los datos necesaria.