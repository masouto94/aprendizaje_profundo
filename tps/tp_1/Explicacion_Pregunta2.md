# TP1 · Pregunta 2, explicada para alguien que nunca vio el tema

**Astorga, González, Souto** · Aprendizaje Profundo, CEIA FIUBA · Cohorte 27

> Este documento no da nada por sabido. Cada palabra técnica aparece en **negrita** la primera vez y se explica ahí mismo.

---

## Abstract

Una tienda guardó 182.721 compras de 7.432 clientes: quién compró, qué producto, de qué categoría y cuánto pagó. Quiere estimar, para cada cliente, **cuánto gasta en promedio por compra**. Sobre todo, quiere poder estimarlo para **clientes nuevos**, que todavía compraron poco.

Primero exploramos los datos y encontramos algo clave: **cada producto tiene un precio fijo**. Por eso no se puede usar el precio de lo que compró el cliente para predecir su gasto: sería darle a la máquina la respuesta escondida. Describimos a cada cliente por **qué tipo de productos elige** (qué parte de sus compras fue de cada categoría) y por sus datos personales (edad, ciudad, etc.).

Después entrenamos una red neuronal. Con el historial completo de cada cliente acierta muy bien: explica el 95% de las diferencias de gasto entre clientes, cuando la consigna pedía al menos el 70%. Pero una fórmula mucho más simple acierta lo mismo, y descubrimos por qué: el gasto promedio es, casi por definición, una suma de "cuánto compra de cada tipo" por "cuánto cuesta cada tipo".

El hallazgo más importante vino al probar el caso que pide el enunciado. **Con clientes nuevos, de 1 a 5 compras, ese modelo fallaba por completo.** Lo resolvimos entrenando a la red también con versiones "recortadas" de los clientes conocidos, como si tuvieran pocas compras. Así aprendió a ser prudente cuando tiene poca información. El modelo resultante sirve para clientes nuevos y, a diferencia de la fórmula simple, sigue funcionando bien con los clientes de historial largo. Ahí es donde la red neuronal se justifica.

---

## Contenido teórico

### 1. Del problema de la Pregunta 1 a este

En la Pregunta 1, la máquina tenía dos perillas y dibujaba una sola curva. Acá es la misma idea, pero más grande: en lugar de una neurona hay muchas, organizadas en capas, y en lugar de un número de entrada hay cientos de datos de cada cliente. Lo demás no cambia. Hay un error que medir, un terreno que bajar y una estrategia para bajarlo (un **optimizador**, como Adam).

### 2. Pasar de compras a clientes

El archivo tiene **una fila por compra**, pero la pregunta es **por cliente**. Así que primero hay que agrupar: juntar todas las compras de cada cliente y resumirlas en una sola fila con números que lo describan. El valor a predecir (el **objetivo**) es el promedio de lo que pagó en todas sus compras.

### 3. Explorar los datos (EDA)

**EDA** quiere decir "análisis exploratorio de datos": mirar los datos antes de modelar. Se revisa qué tipo de dato tiene cada columna, si faltan datos (**nulos**), si hay filas repetidas y qué columnas parecen relacionarse con lo que queremos predecir. Cada decisión posterior tiene que poder justificarse con algo visto en esta etapa.

### 4. Convertir todo en números (encoding)

Una red solo entiende números. La regla es: **si la variable tiene un orden real, se puede usar un número que respete ese orden. Si no lo tiene, no.**

- **Mapping:** reemplazar cada valor por un número elegido a mano ("M" → 1, "F" → 0; "5+" → 5).
- **Ordinal:** numerar categorías que tienen orden, como los rangos de edad.
- **One-hot:** una columna por categoría, con 1 donde corresponde y 0 en las demás. Se usa cuando no hay orden: la categoría de producto 10 no es "más" que la 5.
- **Label encoding:** ponerle un número cualquiera a cada categoría. En una red es peligroso para variables sin orden, porque la red va a creer que 10 está "más cerca" de 9 que de 2.

### 5. Fuga de información (data leakage)

Pasa cuando, sin darse cuenta, a la red se le da **la respuesta escondida dentro de las preguntas**. Los números salen espectaculares, pero el modelo no sirve en la realidad.

Ejemplo de este trabajo: describir a un cliente por "qué parte de su **dinero** gastó en cada categoría" usa lo que pagó, que es justamente lo que queremos predecir. Describirlo por "qué parte de sus **compras** fueron de cada categoría" es legítimo, porque eso se sabe sin mirar los montos.

### 6. Entrenamiento, validación y prueba

Los clientes se reparten en tres grupos:

- **Train** (70%): con estos datos la red aprende.
- **Validation** (15%): se usan para comparar opciones y elegir la mejor.
- **Test** (15%): se guardan hasta el final y se usan **una sola vez**. Como al elegir con validation uno termina "acomodándose" un poco a esos datos, test es la medición honesta.

### 7. La red: un perceptrón multicapa (MLP)

Una **MLP** es un conjunto de neuronas como la de la Pregunta 1, organizadas en **capas**. Cada neurona recibe todos los resultados de la capa anterior, los combina con sus propios pesos y los pasa por una **función de activación**. Las capas del medio se llaman **ocultas**. La última tiene una sola neurona, que da la predicción. La nuestra tiene tres capas ocultas, de 128, 64 y 32 neuronas.

### 8. Funciones de activación

Son las "curvas" que se aplican dentro de cada neurona. Sin ellas, apilar capas no serviría de nada.

- **Tanh y Sigmoid** tienen forma de "S". En la Pregunta 1 vimos su problema: en los extremos son planas y el aprendizaje se frena.
- **ReLU** deja pasar los positivos tal cual y convierte los negativos en 0. No se satura del lado positivo y es la opción por defecto en la mayoría de las redes.
- **LeakyReLU y GELU** son variantes de ReLU.

### 9. Función de costo

Es el error que la red intenta bajar. El más común es el **MSE** (promedio de los errores al cuadrado). La "nota" que pide la consigna, el **R²**, se calcula a partir de él:

$$
R^2 = 1 - \frac{\text{MSE del modelo}}{\text{MSE de predecir siempre el promedio}} \tag{1}
$$

Un R² de 1 es perfecto. Uno de 0 equivale a predecir siempre el promedio y uno negativo es peor que eso. Como el R² se arma con el MSE, entrenar bajando el MSE es entrenar mejorando directamente el R².

### 10. El sobreajuste y cómo frenarlo

Una red con muchos parámetros puede **memorizar** los clientes de train en vez de aprender patrones. Se nota porque el error en train sigue bajando mientras el de validation se estanca o sube. Eso se llama **sobreajuste**. Dos herramientas para frenarlo:

- **Early stopping (parada temprana):** guardar la red en el momento en que validation fue mejor y quedarse con esa versión.
- **Dropout:** durante el entrenamiento, apagar al azar una parte de las neuronas en cada paso, para que la red no dependa de ninguna en particular.

### 11. Aumentar datos (data augmentation)

Consiste en mostrarle a la red **variaciones** de los datos que ya tenemos, para que aprenda situaciones que de otra forma nunca vería. En este trabajo, la variación es: "este mismo cliente, pero como si solo conociéramos 3 de sus compras".

---

## Lo que hicimos

### 2a · Entender los datos y prepararlos

**Primera mirada.** El archivo tiene 11 columnas. Solo dos tienen datos faltantes: las subcategorías del producto (23% y 58% vacías). Hay 2.249 filas idénticas repetidas (1,2%).

**Unidad de análisis.** Los datos personales de cada cliente (edad, género, ciudad, etc.) son siempre los mismos en todas sus compras, así que son atributos del cliente. El objetivo, el gasto promedio, va de 275 a 17.231. Algo llamativo: ningún cliente tiene entre 21 y 29 compras. Están los que compraron de 10 a 20 veces y los que compraron de 30 a 50. Los segundos gastan casi el doble por compra.

**Filas repetidas.** No las borramos. Aparecen mucho más en los clientes que compran más, y si las borráramos, 40 clientes quedarían con cantidades de compras que el archivo nunca tiene. Todo indica que son el mismo producto comprado dos veces, no errores de carga.

**Qué columnas sirven.** La edad, la antigüedad y la ciudad influyen algo en el gasto, pero poco. El tipo de producto lo explica casi todo: la categoría sola explica el 76% del precio de cada compra. Y la columna del producto explica el 100%, porque **cada producto tiene un único precio**. El identificador del cliente y el del producto quedan afuera: uno no significa nada y el otro, usado a través de su precio, sería fuga de información.

**Datos faltantes.** No son errores. Un producto que no tiene subcategoría no la tiene en ninguna venta. Por eso no inventamos un valor: tratamos el vacío como una opción más, "sin subcategoría".

**Encoding.** El género pasa a 0/1. La edad y la antigüedad se numeran en orden. La ciudad también, porque el enunciado dice que A es de clase alta, B media y C media-baja, y los datos lo confirman (A gasta más que B, y B más que C). Las categorías de producto, que no tienen orden, nunca entran como números.

**Separación.** Repartimos los clientes en train, validation y test **antes** de construir cualquier dato, para que nada de lo que aprende el modelo venga de validation o test.

**Cómo describimos a cada cliente.** Con sus datos personales y con **qué parte de sus compras** fue de cada categoría, de cada subcategoría y de cada combinación de categoría y subcategoría. En total, 450 números por cliente. Son proporciones: no usan lo que pagó y no cambian si el cliente compró 10 o 50 veces, como pide el enunciado.

**La prueba de fuga de información.** Medimos qué pasaría con cuatro datos tentadores:

- Con "el precio de lo que compró", el acierto es **perfecto** (R² = 1). No es un logro: es la definición del objetivo.
- Con "qué parte de su dinero gastó en cada categoría", el acierto sube un poco (de 0,961 a 0,969) y parece una mejora. Es fuga de información igual, y los controles de validation no la detectan, porque el problema está dentro de cada cliente.
- La cantidad de compras está muy relacionada con el gasto, pero no agrega nada: los que compran más, compran más productos caros, y eso ya lo dicen sus gustos.

### 2b · La red y cómo la elegimos

**Puntos de comparación.** Antes de ajustar nada medimos dos referencias. Predecir siempre el promedio da R² = 0. Una **regresión lineal** (la fórmula más simple: una suma con pesos de los 450 números) ya llega a **0,961**. La red tiene que justificarse contra eso.

**La red base sobreajusta enseguida.** En la época 3 ya llega a su mejor punto (0,957). Después memoriza: train sube a 0,999 y validation baja.

**Experimentos.** Cambiamos una cosa por vez y repetimos cada prueba con 3 semillas (arranques al azar distintos), para saber si una diferencia es real o casualidad. Regla: solo cambiamos la opción estándar si la alternativa le gana por más que la variación entre semillas.

- **Activación:** ReLU, GELU, LeakyReLU y Sigmoid quedan empatadas. Tanh es claramente la peor y Sigmoid la más lenta, lo mismo que vimos con la saturación en la Pregunta 1. Elegimos ReLU.
- **Función de costo:** MSE, Huber y MAE empatan. Elegimos MSE porque el R² se arma con él.
- **Optimizador y tamaño de paso:** Adam, AdamW y AMSGrad dan prácticamente lo mismo. El tamaño de paso importa más: 0,001 es el mejor. El descenso por gradiente clásico (SGD) con paso chico no llega a aprender en 60 épocas, igual que en la Pregunta 1. Elegimos Adam con 0,001.
- **Dropout:** empeora un poco. La red no necesita esa ayuda porque ya la cortamos a tiempo.
- **Épocas:** el mejor momento es alrededor de la época 5. Usamos early stopping.

**Conclusión de 2b.** Ningún cambio de configuración le ganó a la opción estándar. Lo que más importa en este problema son los datos con los que se describe al cliente y cortar el entrenamiento a tiempo.

### 2c · Evaluación

**Curvas y aciertos.** El modelo final llega a R² **0,957** en validation, con un error promedio de 535 por cliente (10,6%). Acierta un poco mejor en los clientes con más compras.

**Qué aportó cada cosa.** Solo con los datos personales, la red llega a 0,35 (acá sí le gana a la fórmula simple, porque el efecto de la edad sube y después baja). Solo con las categorías llega a 0,944. Las combinaciones categoría-subcategoría la llevan a 0,957.

**La prueba que cambió todo: clientes nuevos.** Simulamos clientes nuevos tomando al azar 1, 3, 5 o 10 compras de cada cliente de validation y le pedimos al modelo que estime su gasto real. Con 10 compras anda razonable (0,74). Con 5, mal (0,29). Con 3 o 1, **peor que no tener modelo** (-0,37 y -3,6). La red nunca había visto un cliente con tan pocas compras y tomaba esas 1 o 3 compras como si fueran todo su historial.

**La solución: entrenar con clientes "recortados".** A cada cliente de train lo mostramos varias veces, como si solo conociéramos 1, 2, 3, 5, 8 o 13 de sus compras, siempre con su gasto real como respuesta. Además usamos menos columnas (las combinaciones casi siempre valen cero con pocas compras) y, ahora sí, un poco de dropout. Resultado: con 1 compra, **0,41**; con 3, **0,54**; con 5, **0,64**; con 10, **0,78**. A cambio, con el historial completo baja de 0,957 a **0,906**.

**Donde la red se gana su lugar.** La fórmula simple entrenada con los mismos recortes se cae a 0,66 con el historial completo. Una suma con pesos fijos no puede "darse cuenta" de cuánta información tiene sobre el cliente. La red sí: reconoce un perfil armado con poca información y modera su respuesta.

### 2d · Prueba final y conclusiones

Con los clientes de test, usados una sola vez, los resultados confirman todo lo anterior. El modelo de 2b llega a 0,953 con historial completo, pero se derrumba con clientes nuevos. El modelo robusto obtiene 0,38 / 0,54 / 0,63 / 0,79 con 1 / 3 / 5 / 10 compras, y 0,91 con el historial completo.

**Recomendación:** usar el modelo robusto, porque el objetivo del enunciado son los clientes nuevos. Si se quiere la máxima precisión con clientes de historial largo, combinar los dos: el robusto mientras el cliente tiene pocas compras y el de 2b cuando ya tiene 10 o más.

---

## Auditoría posterior

### Qué controlamos

| Qué revisamos                                            | Para qué                                                 | Resultado                                                       |
| -------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------- |
| Que los datos personales no cambien dentro de un cliente | Poder tratarlos como atributo del cliente                | Ningún cliente cambia                                           |
| De dónde salen las filas repetidas                       | No borrar compras reales                                 | Son recompras: borrarlas crea cantidades imposibles             |
| Que los nulos sean estructurales                         | No inventar subcategorías                                | Ningún producto mezcla filas con y sin subcategoría             |
| Separar por cliente antes de construir los datos         | Que nada de validation o test se filtre al entrenamiento | Vocabulario y escalado salen solo de train                      |
| Probar las features tentadoras                           | Detectar fuga de información con números                 | Precio por producto: R² = 1; proporción del gasto: mejora falsa |
| 3 semillas por experimento                               | Distinguir mejoras reales del azar                       | Ningún hiperparámetro superó el ruido                           |
| Test usado una sola vez                                  | Medición honesta                                         | 0,953 en test contra 0,957 en validation                        |
| Simulación de clientes nuevos                            | Probar el caso de uso del enunciado                      | Detectamos la falla y la corregimos                             |

### Preguntas que nos podrían hacer

**¿Por qué no usaron el precio de los productos, si da R² = 1?**
Porque cada producto tiene un único precio, entonces promediar los precios de lo que compró el cliente **es** su gasto promedio. No sería predecir, sería calcular el resultado con los mismos datos que lo definen.

**Si una regresión lineal da lo mismo, ¿para qué la red?**
Con el historial completo, la red no hace falta, y lo decimos. Pero para clientes nuevos, que son el objetivo del enunciado, la fórmula simple no puede adaptar su prudencia a la cantidad de información y la red sí (0,91 contra 0,66 con historial completo, 0,78 contra 0,63 con 10 compras).

**¿Por qué no eligieron GELU o Huber, que dieron un poco mejor?**
Porque la diferencia fue menor que la variación entre semillas. Elegir por esa diferencia sería elegir por azar. En esos casos nos quedamos con la opción estándar.

**¿Por qué la red se entrena tan pocas épocas?**
Porque a partir de la época 3 a 5 empieza a memorizar a los clientes de train. Las curvas lo muestran: train sigue mejorando y validation empeora.

**¿Cómo simularon clientes nuevos si no hay fechas?**
Elegimos compras al azar de cada cliente. Con fechas habría sido más realista usar sus primeras compras. Lo dejamos anotado como limitación.

### Fuentes

- Kingma y Ba (2015), *Adam: A Method for Stochastic Optimization*: https://arxiv.org/abs/1412.6980
- Reddi, Kale y Kumar (2018), *On the Convergence of Adam and Beyond* (AMSGrad): https://openreview.net/forum?id=ryQu7f-RZ
- Ruder (2016), *An overview of gradient descent optimization algorithms*: https://arxiv.org/abs/1609.04747
- Loshchilov y Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW): https://arxiv.org/abs/1711.05101
- Material de la materia: https://github.com/FIUBA-Posgrado-Inteligencia-Artificial/aprendizaje_profundo
